package com.example.ss0327;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class Link_page extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_link_page);
        Button link=(Button) findViewById(R.id.linkbtn);
        link.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //start_page로 다시 돌아감
                onBackPressed();
            }
        });

    }
}