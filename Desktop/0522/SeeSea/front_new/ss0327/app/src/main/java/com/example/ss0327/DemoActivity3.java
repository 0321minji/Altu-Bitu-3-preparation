//
//package com.example.ss0327;
//import android.app.ProgressDialog;
//import android.os.Bundle;
//
//import androidx.appcompat.app.AppCompatActivity;
//
//import com.google.android.exoplayer2.MediaItem;
//import com.google.android.exoplayer2.SimpleExoPlayer;
//import com.google.android.exoplayer2.ui.PlayerView;
//
//public class DemoActivity extends AppCompatActivity {
//
//    private String websiteUri = "http://clips.vorwaerts-gmbh.de/big_buck_bunny.mp4";
//    private ProgressDialog progressDialog;
//
//    PlayerView playerView;
//    SimpleExoPlayer simpleExoPlayer;
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_demo);
//
//        playerView = findViewById(R.id.playerView);
//        progressDialog = new ProgressDialog(this);
//        progressDialog.setMessage("buffering");
//        progressDialog.setCancelable(true);
//        playVideo();
//    }
//    private void playVideo(){
//        try {
//            simpleExoPlayer = new SimpleExoPlayer.Builder(this).build();
//            playerView.setPlayer(simpleExoPlayer);
//            MediaItem mediaItem = MediaItem.fromUri(websiteUri);
//            simpleExoPlayer.addMediaItem(mediaItem);
//            simpleExoPlayer.prepare();
//            simpleExoPlayer.play();
//
//        }catch (Exception e)
//        {
//            progressDialog.dismiss();
//        }
//    }
//    @Override
//    public void onBackPressed()
//    {
//        super.onBackPressed();
//        simpleExoPlayer.pause();
//    }
//    @Override
//    protected void onDestroy()
//    {
//        super.onDestroy();
//        simpleExoPlayer.pause();
//    }
//
//}