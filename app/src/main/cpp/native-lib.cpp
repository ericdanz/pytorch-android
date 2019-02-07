#include <jni.h>
#include <string>
#include <algorithm>
#define PROTOBUF_USE_DLLS 1
#define CAFFE2_USE_LITE_PROTO 1
#include <unistd.h>
#include <caffe2/predictor/predictor.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/timer.h>

#include "caffe2/core/init.h"
#include <caffe2/core/tensor.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <ATen/ATen.h>
#include "classes.h"
#define IMG_H 270
#define IMG_W 480
//#define IMG_H 227
//#define IMG_W 227
//#define IMG_H 270
//#define IMG_W 270
#define IMG_C 3
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_C
#define alog(...) __android_log_print(ANDROID_LOG_ERROR, "F8DEMO", __VA_ARGS__);

static caffe2::NetDef _initNet, _predictNet;
static caffe2::Predictor *_predictor;
static char raw_data[MAX_DATA_SIZE];
static float input_data[MAX_DATA_SIZE];
static caffe2::Workspace ws;

// A function to load the NetDefs from protobufs.
void loadToNetDef(AAssetManager* mgr, caffe2::NetDef* net, const char *filename) {
    AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    assert(asset != nullptr);
    const void *data = AAsset_getBuffer(asset);
    assert(data != nullptr);
    off_t len = AAsset_getLength(asset);
    assert(len != 0);
    if (!net->ParseFromArray(data, len)) {
        alog("Couldn't parse net from data.\n");
    }
    AAsset_close(asset);
}

extern "C"
void
Java_facebook_f8demo_ClassifyCamera_initCaffe2(
        JNIEnv* env,
        jobject /* this */,
        jobject assetManager) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    alog("Attempting to load protobuf netdefs...");
//    loadToNetDef(mgr, &_initNet,   "resnet18_init_net_v1.pb");
//    loadToNetDef(mgr, &_predictNet,"resnet18_predict_net_v1.pb");
    loadToNetDef(mgr, &_initNet,   "shuf_esp2_kp_init_net_v1.pb");
    loadToNetDef(mgr, &_predictNet,"shuf_esp2_kp_predict_net_v1.pb");
    alog("done.");
    alog("Instantiating predictor...");
    _predictor = new caffe2::Predictor(_initNet, _predictNet);
    alog("done.")
}

float avg_fps = 0.0;
float total_fps = 0.0;
int iters_fps = 10;

extern "C"
JNIEXPORT jstring JNICALL
Java_facebook_f8demo_ClassifyCamera_classificationFromCaffe2(
        JNIEnv *env,
        jobject /* this */,
        jint h, jint w, jbyteArray Y, jbyteArray U, jbyteArray V,
        jint rowStride, jint pixelStride,
        jboolean infer_HWC) {
    if (!_predictor) {
        return env->NewStringUTF("Loading...");
    }
    jsize Y_len = env->GetArrayLength(Y);
    jbyte * Y_data = env->GetByteArrayElements(Y, 0);
    assert(Y_len <= MAX_DATA_SIZE);
    jsize U_len = env->GetArrayLength(U);
    jbyte * U_data = env->GetByteArrayElements(U, 0);
    assert(U_len <= MAX_DATA_SIZE);
    jsize V_len = env->GetArrayLength(V);
    jbyte * V_data = env->GetByteArrayElements(V, 0);
    assert(V_len <= MAX_DATA_SIZE);

#define min(a,b) ((a) > (b)) ? (b) : (a)
#define max(a,b) ((a) > (b)) ? (a) : (b)

    auto h_offset = max(0, (h - IMG_H) / 2);
    auto w_offset = max(0, (w - IMG_W) / 2);

    auto iter_h = IMG_H;
    auto iter_w = IMG_W;
    if (h < IMG_H) {
        iter_h = h;
    }
    if (w < IMG_W) {
        iter_w = w;
    }
//    alog("before data shaping %d %d",iter_h,iter_w);
//    alog("%d %d",h_offset, w_offset);
//    alog("%d y %d u %d v %d",pixelStride,Y_len, U_len, V_len);

    float b_mean = 104.00698793f;
    float g_mean = 116.66876762f;
    float r_mean = 122.67891434f;
    for (auto i = 0; i < iter_h; ++i) {
        jbyte* Y_row = &Y_data[(h_offset + i) * w];
        jbyte* U_row = &U_data[(h_offset + i) / 2 * rowStride];
        jbyte* V_row = &V_data[(h_offset + i) / 2 * rowStride];
        for (auto j = 0; j < iter_w; ++j) {
            // Tested on Pixel and S7.
            char y = Y_row[w_offset + j];
            char u = U_row[pixelStride * ((w_offset+j)/pixelStride)];
            char v = V_row[pixelStride * ((w_offset+j)/pixelStride)];


            auto b_i = 0 * IMG_H * IMG_W + i * IMG_W + j;
            auto g_i = 1 * IMG_H * IMG_W + i * IMG_W + j;
            auto r_i = 2 * IMG_H * IMG_W + i * IMG_W + j;

            if (infer_HWC) {
                b_i = (j * IMG_W + i) * IMG_C;
                g_i = (j * IMG_W + i) * IMG_C + 1;
                r_i = (j * IMG_W + i) * IMG_C + 2;
            }
/*
  R = Y + 1.402 (V-128)
  G = Y - 0.34414 (U-128) - 0.71414 (V-128)
  B = Y + 1.772 (U-V)
 */

            input_data[r_i] = -r_mean + (float) ((float) min(255., max(0., (float) (y + 1.402 * (v - 128)))));
            input_data[g_i] = -g_mean + (float) ((float) min(255., max(0., (float) (y - 0.34414 * (u - 128) - 0.71414 * (v - 128)))));
            input_data[b_i] = -b_mean + (float) ((float) min(255., max(0., (float) (y + 1.772 * (u - v)))));


        }
    }
//    alog("before input");
    caffe2::TensorCPU input;
    if (infer_HWC) {
        input = caffe2::Tensor(std::vector<int>({IMG_H, IMG_W, IMG_C}), caffe2::CPU);
    } else {
        input = caffe2::Tensor(std::vector<int>({1, IMG_C, IMG_H, IMG_W}), caffe2::CPU);
    }
    memcpy(input.mutable_data<float>(), input_data, IMG_H * IMG_W * IMG_C * sizeof(float));
    caffe2::TensorCPU out = caffe2::Tensor(std::vector<int>({1, 28, 67, 120}), caffe2::CPU);
    std::vector<caffe2::TensorCPU> input_vec({input});
    std::vector<caffe2::TensorCPU> output_vec({out});
    caffe2::Timer t;
    t.Start();
    (*_predictor)(input_vec, &output_vec);
    float fps = 1000/t.MilliSeconds();
    total_fps += fps;
    avg_fps = total_fps / iters_fps;
    total_fps -= avg_fps;
    std::ostringstream stringStream;
    stringStream << avg_fps << " FPS\n" << output_vec.size();
//    int ii = 0;
//    for (auto output : output_vec) {
//        auto data = output.data<float>();
//        ii = ii + 1;
//    }
//    stringStream << " " << ii ;
    std::ostringstream keypoints_stream;
//    keypoints_stream << output_vec[0].size() << " ";
    auto out_v = output_vec[0].data<float>();
//    keypoints_stream << out_v.size();
//    int index = 0 + 25*67*12 + 50*120 + 50;

//    keypoints_stream << out_v[index];
//    return env->NewStringUTF(keypoints_stream.str().c_str());

    for (auto i = 0; i < 67; ++i)
        for (auto j = 0; j < 120; ++j)
        {
            int max_index = 25;
            float max_val = 0;
            //find the max in depth
            for (auto k = 0; k < 26; ++k)
            {
                int index =  k * 67 * 120 + i * 120 + j;
                float val = out_v[index];
                if (val > max_val)
                {
                    max_val = val;
                    max_index = k;
                }
            }

            //if its bigger than neighbors in x and y, keep it
            if (max_val > 0 && max_index != 25)
            {
                bool stop_checking = false;
                for (auto io = i-2; io < i+4; ++io)
                {
                    for (auto jo = j-2; jo < j+4; ++jo)
                    {
                        if (jo == j and io == i and stop_checking) continue;
                        //subtract 2 from io and jo to iterate over indices
                        int index = max_index * 67 * 120 + io * 120 + jo;
                        float check_val = out_v[index];
                        if (check_val > max_val) {
                            max_val = 0;
                            max_index = 25;
                            stop_checking = true;
                        }

                    }
                }
                if (!stop_checking){
                    keypoints_stream << " " << max_index << "," << i << "," << j;
                }
            }

        }
        //keypoints_stream

//    constexpr int k = 5;
//    float max[k] = {0};
//    int max_index[k] = {0};
//    // Find the top-k results manually.
//    for (auto output : output_vec) {
//        auto data = output.data<float>();
//        for (auto i = 0; i < output.size(); ++i) {
//            for (auto j = 0; j < k; ++j) {
//                if (data[i] > max[j]) {
//                    for (auto _j = k - 1; _j > j; --_j) {
//                        max[_j - 1] = max[_j];
//                        max_index[_j - 1] = max_index[_j];
//                    }
//                    max[j] = data[i];
//                    max_index[j] = i;
//                    goto skip;
//                }
//            }
//            skip:;
//        }
//    }


//    for (auto j = 0; j < k; ++j) {
//        stringStream << j << ": " << imagenet_classes[max_index[j]] << " - " << max[j] / 10 << "%\n";
//    }
    return env->NewStringUTF(keypoints_stream.str().c_str());
}
