package com.example.ss0327;


import android.app.Dialog;
import android.content.Context;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.media.SoundPool;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.MediaController;
import android.widget.TextView;
import android.widget.VideoView;


import androidx.annotation.NonNull;

import org.w3c.dom.Text;

public class alarm extends Dialog {
    MediaPlayer mediaPlayer;
    alarm Alarm;
    VideoView videoView;

    public alarm(Context context){

        super(context, android.R.style.Theme_Translucent_NoTitleBar);
    }
    public alarm(){
        super(null);
    }
    @Override
    public void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        WindowManager.LayoutParams IpWindow = new WindowManager.LayoutParams();
        IpWindow.flags=WindowManager.LayoutParams.FLAG_DIM_BEHIND;
        IpWindow.dimAmount=0.5f;
        getWindow().setAttributes(IpWindow);
        TextView dist=(TextView) findViewById(R.id.textView2);



        /*if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            soundPool = new SoundPool.Builder().build();
        }
        soundManager = new SoundManager(soundPool);
        soundManager.addSound(0,R.raw.siren_sound);*/
        setContentView(R.layout.activity_alarm);

        mediaPlayer = MediaPlayer.create(getContext(),R.raw.siren_sound);
        mediaPlayer.setLooping(true);
        Alarm=this;
        //soundManager.playSound(0);
        TextView tv=(TextView) this.findViewById(R.id.textView);
        tv.setText("!!Warning!!");
        TextView tv2=(TextView) this.findViewById(R.id.textView2);
        videoView = findViewById(R.id.videoView3);
        String videoPath = "android.resource://" + getContext().getPackageName() + "/" + R.raw.data2;
        //String videoPath = "android.resource://" + getContext().getPackageName() + "/" + R.raw.my_final;

        //Replace video_file with the name of your video file in the res/raw directory
        videoView.setVideoURI(Uri.parse(videoPath));
        videoView.setRotation(180);
        /*MediaController mediaController = new MediaController(this);
        videoView.setMediaController(mediaController);*/

        videoView.start();
        Button bt=(Button) this.findViewById(R.id.btn_shutdown);
        bt.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                onClickBtn(v);
            }
        });

        mediaPlayer.start();
    }
    public void dismiss(){
        super.dismiss();
        mediaPlayer.release();
    }
    public void onClickBtn(View _oView){
        this.dismiss();
    }
}
