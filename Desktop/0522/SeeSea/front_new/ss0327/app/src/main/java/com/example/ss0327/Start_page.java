package com.example.ss0327;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import retrofit2.Retrofit;

public class Start_page extends AppCompatActivity {
    private Retrofit mRetrofit;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start_page);
        Button setbtn = (Button) findViewById(R.id.setbtn);
        //Button linkbtn = (Button) findViewById(R.id.linkbtn);
        Button endbtn = (Button) findViewById(R.id.endbtn);
        Button startbtn = (Button) findViewById(R.id.startbtn);
        // 데모 버튼 이거임
        Button startdemo = (Button) findViewById(R.id.startdemo);
        //setRetrofitInit();
        setbtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(getApplicationContext(), SettingActivity.class);
                startActivity(intent);
            }
        });
        //demo 버튼 누르면 데모 액티비티~
        startdemo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(getApplicationContext(), DemoActivity.class);
                startActivity(intent);
            }
        });
        /*
        linkbtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(getApplicationContext(), Link_page.class);
                startActivity(intent);
            }
        });*/
        endbtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                finishAffinity();
                System.runFinalization();
                System.exit(0);
            }
        });
        startbtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //감시 시작 버튼 -> mainactivity로 넘어감
                Intent intent=new Intent(getApplicationContext(),MainActivity2.class);
                startActivity(intent);
                /*Intent intent=new Intent(getApplicationContext(),fullscreen.class);
                startActivity(intent);*/
                // api 받아오기
                //connectServer();
                // api 받아오기 ver2
                //callFlaskApi/();
                /*String url = "http://10.0.2.2:5000/predict";
                HttpRequestTask requestTask = new HttpRequestTask();
                requestTask.execut e(url);*/
            }
        });
    }
}