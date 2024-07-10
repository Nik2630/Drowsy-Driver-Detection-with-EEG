/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "arduinoFFT.h"
#include "eeg_model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 150* 1024;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


#define SAMPLE_RATE 128
#define BAUD_RATE 115200
#define INPUT_PIN0 A0
#define INPUT_PIN1 A1

#define SAMPLES (SAMPLE_RATE * 2) // Number of samples in 2 second
#define BUZZER_PIN 9

// Function to control the buzzer if drowsiness is detected
void controlBuzzer(int output) {
  if (output == 2) {
    digitalWrite(BUZZER_PIN, HIGH);
  } else if (output == 1) {
    digitalWrite(BUZZER_PIN, HIGH); 
    delay(600); 
    digitalWrite(BUZZER_PIN, LOW); 
    delay(600); 
  } else {
    digitalWrite(BUZZER_PIN, LOW);
  }
}

// Define the FFT object
arduinoFFT FFT = arduinoFFT();

float mean[72]={-1.53990453e+01, -1.35576086e+01, -1.49224892e+01, -1.66449935e+01,
       -1.75362325e+01, -1.79279531e+01, -1.80131138e+01, -1.81042549e+01,
       -1.80288324e+01, -1.82766332e+01, -1.87636080e+01, -1.90415617e+01,
       -1.91702216e+01, -1.91837769e+01, -1.92272055e+01, -1.92857535e+01,
       -1.94282631e+01, -1.95427810e+01, -1.96273866e+01, -1.97988076e+01,
       -1.99236837e+01, -1.99111513e+01, -1.98737467e+01, -1.99433244e+01,
       -2.01226743e+01, -2.02495807e+01, -2.03486190e+01, -2.05267503e+01,
       -2.06831136e+01, -2.08068955e+01, -2.09534856e+01, -2.10648041e+01,
       -2.11336487e+01, -2.12492105e+01, -2.13923746e+01, -2.15555380e+01,
        1.61791754e+01,  1.64426496e+01,  1.19306306e+01,  7.03333703e+00,
        4.26483317e+00,  2.28575132e+00,  1.24757257e+00,  3.18587492e-01,
       -4.65216358e-01, -8.97580684e-01, -1.45775450e+00, -1.98237712e+00,
       -2.16773096e+00, -2.18883779e+00, -2.17853048e+00, -1.88726162e+00,
       -1.39752778e+00, -1.01007965e+00, -6.64272578e-01, -2.59147384e-01,
        2.86580048e-02, -1.03333209e-02, -5.33461669e-01, -1.48548248e+00,
       -2.63581160e+00, -3.63090371e+00, -4.48237239e+00, -5.33935345e+00,
       -6.02881122e+00, -6.59193087e+00, -7.00033716e+00, -7.30759783e+00,
       -7.63795435e+00, -7.96233603e+00, -8.33590692e+00, -8.77340311e+00};
      
float vari[72]={7.38196321, 6.77700687, 6.2769426 , 5.79641363, 5.48472281,
       5.1996816 , 4.84937092, 4.59355365, 4.32949692, 4.22136737,
       4.17993803, 4.08791846, 4.00264181, 3.97906632, 3.98940049,
       3.89681081, 3.86582735, 3.71781324, 3.66067064, 3.66414553,
       3.62821089, 3.5946615 , 3.59274436, 3.53546362, 3.51580045,
       3.53876963, 3.42222668, 3.38862749, 3.34043046, 3.30496878,
       3.32072052, 3.34937902, 3.37684739, 3.39197018, 3.3592207 ,
       3.38801713, 9.23590263, 8.20026559, 6.40708505, 4.37888186,
       3.82452278, 3.52566728, 3.49045596, 3.35357194, 3.30363229,
       3.33687996, 3.37917304, 3.36381617, 3.38199728, 3.41860226,
       3.62728458, 3.97246857, 4.39313116, 4.70408796, 5.01151906,
       5.59974879, 5.9175877 , 5.88136075, 5.56798114, 5.07079736,
       4.53336762, 3.99281666, 3.56198309, 3.16076902, 2.90121396,
       2.81209985, 2.85040891, 2.82788734, 2.94923015, 3.06985531,
       3.06512387, 3.02810292};
// The name of this function is important for Arduino compatibility.
void setup() {
  pinMode(BUZZER_PIN, OUTPUT);

  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(eeg_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  // Available ops:
  //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
  // static tflite::MicroMutableOpResolver<2> micro_op_resolver;;
  // micro_op_resolver.AddFullyConnected();
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;

  // Serial connection begin
  Serial.begin(BAUD_RATE);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Calculate an x value to feed into the model.
  // Calculate elapsed time
  unsigned long reset = micros();
  
  static unsigned long past = 0;
  int sample_index = 0;

  // Create an array to store the input signal samples
  static double samples0[SAMPLES*8];
  static double samples1[SAMPLES*8];
  float x_in[8][72];
  while (sample_index<=SAMPLES*8) {
    unsigned long present = micros();
    unsigned long interval = present - past;
    past = present;

    // Run timer
    static long timer = 0;
    timer -= interval;

    // Sample
    if(timer < 0){
      timer += 1000000 / SAMPLE_RATE;
      float sensor_value0 = analogRead(INPUT_PIN0);
      float sensor_value1 = analogRead(INPUT_PIN1);
      float signal0 = EEGFilter(sensor_value0);
      float signal1 = EEGFilter(sensor_value1);
      Serial.print(signal0);
      Serial.print(",");
      Serial.println(signal1);

      // Store the signal in the samples array
      samples0[sample_index] = signal0;
      samples1[sample_index] = signal1;
      sample_index++;
  }

  // Perform FFT when the desired time interval is reached
  for (int a=0;a<8;a++) {
    double samples0_256[SAMPLES];
    double samples1_256[SAMPLES];
    for (int j=0;j<SAMPLES;j++){
      samples0_256[j]=samples0[j+a*SAMPLES];
      samples1_256[j]=samples1[j+a*SAMPLES];
    }
    // Perform FFT on the samples
    double magnitudes0[SAMPLES];
    double vReal0[SAMPLES];
    double vImag0[SAMPLES];
    FFT.Windowing(samples0_256, SAMPLES, FFT_WIN_TYP_BLACKMAN,FFT_FORWARD); // Apply a Blackman window
    FFT.Compute(vReal0, vImag0, SAMPLES, FFT_FORWARD);           // Perform the FFT
    FFT.ComplexToMagnitude(samples0_256, magnitudes0, SAMPLES); // Get the magnitudes
    // Perform FFT on the samples
    double magnitudes1[SAMPLES];
    double vReal1[SAMPLES];
    double vImag1[SAMPLES];
    FFT.Windowing(samples1_256, SAMPLES, FFT_WIN_TYP_BLACKMAN,FFT_FORWARD); // Apply a Blackman window
    FFT.Compute(vReal1, vImag1, SAMPLES, FFT_FORWARD);           // Perform the FFT
    FFT.ComplexToMagnitude(samples1_256, magnitudes1, SAMPLES); // Get the magnitudes

    float fft_mag[72];

    // Print the frequency and magnitude values
    for (int i = 0; i < 36; i++) {
      //double frequency = i * SAMPLE_RATE / SAMPLES;
      fft_mag[i]=magnitudes0[i];
    }
    for (int i = 0; i < 36; i++) {
      //double frequency = i * SAMPLE_RATE / SAMPLES;
      fft_mag[i+36]=magnitudes1[i];
    }

    for (int i=0;i<72;i++){
      x_in[a][i]=fft_mag[i];
    }
  } 

  float x_avg[72];
  for (int i=0;i<72;i++) {
    float sum=0;
    for (int j=0;j<8;j++) {
      sum += x_in[j][i];
    }
    x_avg[i]=sum/8;
  }
  float x_scaled[72];
  for (int i; i<72;i++){
    x_scaled[i]=(x_avg[i]-mean[i])/vari[i];
  }
//        // Quantize the input from floating-point to integer
//        int8_t x_quantized = x / input->params.scale + input->params.zero_point;
//        // Place the quantized input in the model's input tensor
//        input->data.int8[0] = x_quantized;
//      
//        // Run inference, and report any error
//        TfLiteStatus invoke_status = interpreter->Invoke();
//        if (invoke_status != kTfLiteOk) {
//          MicroPrintf("Invoke failed on x: %f\n", static_cast<double>(x));
//          return;
//        }
//      
//        // Obtain the quantized output from model's output tensor
//        int8_t y_quantized = output->data.int8[0];
//        // Dequantize the output from integer to floating-point
//        float y = (y_quantized - output->params.zero_point) * output->params.scale;

  for (int i = 0; i < 72; i++) {
    input->data.f[i] = x_scaled[i];
  }

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }


  float max_score;
  int max_index;
  for (int i = 0; i < 3; i++) {
    const float score = output->data.f[i];
    // Serial.println(score);
    if ((i == 0) || (score > max_score)) {
      max_score = score;
      max_index = i;
    }
  }
  
  int y=max_index;
  controlBuzzer(y);
  //Serial.println(y);
  
//        MicroPrintf("Found %s (%d.%d%%)", labels[max_index],
//                    static_cast<int>(max_score_int),
//                    static_cast<int>(max_score_frac * 100));

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
//        HandleOutput(x, y);
  }

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
//  inference_count += 1;
//  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}//loop

// Band-Pass Butterworth IIR digital filter, generated using filter_gen.py.
// Sampling rate: 256.0 Hz, frequency: [0.5, 29.5] Hz.
// Filter is order 4, implemented as second-order sections (biquads).
// Reference: 
// https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
// https://courses.ideate.cmu.edu/16-223/f2020/Arduino/FilterDemos/filter_gen.py
float EEGFilter(float input) {
  float output = input;
  {
    static float z1, z2; // filter section state
    float x = output - -0.95391350*z1 - 0.25311356*z2;
    output = 0.00735282*x + 0.01470564*z1 + 0.00735282*z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2; // filter section state
    float x = output - -1.20596630*z1 - 0.60558332*z2;
    output = 1.00000000*x + 2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2; // filter section state
    float x = output - -1.97690645*z1 - 0.97706395*z2;
    output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2; // filter section state
    float x = output - -1.99071687*z1 - 0.99086813*z2;
    output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  return output;
}
