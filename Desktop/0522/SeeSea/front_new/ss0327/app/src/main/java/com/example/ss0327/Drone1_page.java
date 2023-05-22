package com.example.ss0327
        ;

import androidx.appcompat.app.AppCompatActivity;
import java.lang.Math;
import android.content.Intent;
import android.os.Bundle;
import android.widget.TextView;

public class Drone1_page extends AppCompatActivity {
    //ArrayList<String> data=new ArrayList<String>();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_drone1_page);
        Intent intent= getIntent();
        int width = 50;
        int height = 40;

        int x = (int) Math.ceil(width / 11.88);
        int y = (int) Math.ceil(height / 24.807);

        //data = (ArrayList<String>) intent.getSerializableExtra("range");
        //
        Range range=(Range)intent.getSerializableExtra("range");
        // 나중에 수식 계산 할려면

        //Integer x_range=Integer.parse(range.x_range);
        Double x_range=Double.parseDouble(range.x_range);
        Double y_range=Double.parseDouble(range.y_range);
        x_range/=11.88;
        y_range/=24.807;
        Double dr=Math.ceil(2 * x_range * y_range);
        int dr2=(int)Math.round(dr);
        TextView tv = (TextView)findViewById(R.id.tvtest);
        TextView tv2 = (TextView)findViewById(R.id.tvtest2);
        TextView nt=(TextView) findViewById(R.id.dronenotice);
        nt.setText("해당 범위를 감시하기 위해\n 최소 "+String.valueOf(dr2)+"대의 드론이 필요합니다.\n\n 드론을 연결하려면 \n 아래 버튼을 눌러주세요.");
        String xx=range.x_range;
        String yy=range.y_range;

        tv.setText(xx);
        tv2.setText(yy);
    }

}