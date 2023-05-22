//package com.example.ss0327;
//
//import android.os.Bundle;
//import android.widget.ImageView;
//
//import androidx.appcompat.app.AppCompatActivity;
//
//import com.bumptech.glide.Glide;
//
//import java.io.IOException;
//
//import okhttp3.Call;
//import okhttp3.Callback;
//import okhttp3.OkHttpClient;
//import okhttp3.Request;
//import okhttp3.Response;
//
//public class DemoActivity extends AppCompatActivity {
//    private ImageView imageView;
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_demo);
//
//        imageView = findViewById(R.id.imageView3);
//
//        Request request = new Request.Builder()
//                .url("http://10.0.2.2:5000/video_feed") // Flask 서버의 주소로 변경
//                .build();
//
//        OkHttpClient client = new OkHttpClient();
//
//        client.newCall(request).enqueue(new Callback() {
//            @Override
//            public void onFailure(Call call, IOException e) {
//                e.printStackTrace();
//            }
//
//            @Override
//            public void onResponse(Call call, Response response) throws IOException {
//                final byte[] imageData = response.body().bytes();
//                runOnUiThread(new Runnable() {
//                    @Override
//                    public void run() {
//                        Glide.with(getApplicationContext())
//                                .load(imageData)
//                                .into(imageView);
//                    }
//                });
//            }
//        });
//    }
//}