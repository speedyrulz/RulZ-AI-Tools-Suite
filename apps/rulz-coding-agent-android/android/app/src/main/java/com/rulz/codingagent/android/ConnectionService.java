package com.rulz.codingagent.android;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.net.Uri;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;
import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;

/**
 * Keeps the app's process (and so its connection to the computer) alive while the app is in the
 * background, so it can tell the owner when a task finishes or needs them. Android requires the
 * ongoing notification this shows; it also shows what the agent is doing right now.
 */
public class ConnectionService extends Service {

    static final String CHANNEL = "status";
    static final String CHANNEL_WAITING = "waiting";
    static final String CHANNEL_FINISHED = "finished";
    static final int NOTIFICATION_ID = 7420;
    static final String EXTRA_TITLE = "title";
    static final String EXTRA_TEXT = "text";
    /** A task can't keep the phone awake longer than this. */
    private static final long MAX_BUSY_MS = 3 * 60 * 60 * 1000L;

    private static ConnectionService running;
    private static boolean busy;
    private static String title = "Connected";
    private static String text = "";
    private PowerManager.WakeLock wakeLock;

    static void createChannels(Context context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return;
        NotificationManager nm = context.getSystemService(NotificationManager.class);
        // 0.2.0 showed a silent "connection" channel at the lowest importance, which can't be raised.
        nm.deleteNotificationChannel("connection");
        NotificationChannel connection = new NotificationChannel(CHANNEL, "What the agent is doing", NotificationManager.IMPORTANCE_LOW);
        connection.setDescription("What the agent is doing, while RulZ Agent stays connected in the background");
        connection.setShowBadge(false);
        nm.createNotificationChannel(connection);
        // Same ids as earlier versions used, so the owner's choices for them are kept.
        NotificationChannel waiting = new NotificationChannel(CHANNEL_WAITING, "Waiting for you", NotificationManager.IMPORTANCE_HIGH);
        waiting.setDescription("Permission requests, questions and pauses");
        waiting.enableVibration(true);
        nm.createNotificationChannel(waiting);
        NotificationChannel finished = new NotificationChannel(CHANNEL_FINISHED, "Task finished", NotificationManager.IMPORTANCE_DEFAULT);
        finished.setDescription("A task finished on your computer");
        nm.createNotificationChannel(finished);
    }

    /** Opens the app; `what` tells it where to go (rulzagent://open/<what>). */
    static PendingIntent openApp(Context context, String what, int requestCode) {
        Intent open = new Intent(context, MainActivity.class)
            .setAction(Intent.ACTION_VIEW)
            .setData(Uri.parse("rulzagent://open/" + what))
            .setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        return PendingIntent.getActivity(context, requestCode, open, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
    }

    private static Notification build(Context context) {
        return new NotificationCompat.Builder(context, CHANNEL)
            .setSmallIcon(R.drawable.ic_stat_rulz)
            .setColor(0xFFED000D)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(new NotificationCompat.BigTextStyle().bigText(text))
            .setContentIntent(openApp(context, "project", 0))
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build();
    }

    /** Changes what the ongoing notification says. Works while the app is in the background. */
    static synchronized boolean update(Context context, String newTitle, String newText) {
        title = newTitle;
        text = newText;
        if (running == null) return false;
        try {
            NotificationManagerCompat.from(context).notify(NOTIFICATION_ID, build(running));
        } catch (SecurityException e) {
            // No permission to show notifications; the service keeps running anyway.
        }
        return true;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && intent.hasExtra(EXTRA_TITLE)) title = intent.getStringExtra(EXTRA_TITLE);
        if (intent != null && intent.hasExtra(EXTRA_TEXT)) text = intent.getStringExtra(EXTRA_TEXT);
        createChannels(this);
        Notification notification = build(this);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        } else {
            startForeground(NOTIFICATION_ID, notification);
        }
        running = this;
        applyBusy();
        return START_NOT_STICKY;
    }

    /** While a task runs on the computer, the phone stays awake so its messages arrive right away. */
    static synchronized void setBusy(boolean value) {
        busy = value;
        if (running != null) running.applyBusy();
    }

    private synchronized void applyBusy() {
        if (busy) {
            if (wakeLock == null) {
                PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
                wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "RulZAgent:task");
                wakeLock.setReferenceCounted(false);
            }
            wakeLock.acquire(MAX_BUSY_MS);
        } else if (wakeLock != null && wakeLock.isHeld()) {
            wakeLock.release();
        }
    }

    @Override
    public void onDestroy() {
        running = null;
        if (wakeLock != null && wakeLock.isHeld()) wakeLock.release();
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
