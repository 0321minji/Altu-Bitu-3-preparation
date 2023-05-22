package com.example.ss0327;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;


public class SettingActivity extends AppCompatActivity {
    Button btn_main;
    EditText r1;
    EditText r2;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_setting_activity);
        btn_main=(Button) findViewById(R.id.button_main);
        btn_main.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                r1=(EditText) findViewById(R.id.range1);
                r2=(EditText) findViewById(R.id.range2);
                String x_range=r1.getText().toString();
                String y_range=r2.getText().toString();
                Range range=new Range(x_range,y_range);
                Intent intent=new Intent (getApplicationContext(),Drone1_page.class);
                intent.putExtra("range",range);
                startActivity(intent);
                //start page로 돌아감
                //onBackPressed();
                /*
                Intent intent= new Intent(getApplicationContext(),MainActivity2.class);
                startActivity(intent);*/

                //startActivity(myIntent);
            }
        });

    }
}
