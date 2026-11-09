package com.rulz.codingagent.android;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.PowerManager;
import android.provider.Settings;
import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;
import androidx.core.content.ContextCompat;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

/**
 * The background connection (ConnectionService), the app's notifications, and the Android settings
 * pages they depend on. Notifications are posted here directly: they must work while the app is in
 * the background, where nothing may be asked of the owner first.
 */
@CapacitorPlugin(name = "KeepAlive")
public class KeepAlivePlugin extends Plugin {

    @Override
    public void load() {
        ConnectionService.createChannels(getContext());
    }

    /** Starts the background connection: { title, text }. Only allowed while the app is on screen. */
    @PluginMethod
    public void start(PluginCall call) {
        Intent intent = new Intent(getContext(), ConnectionService.class)
            .putExtra(ConnectionService.EXTRA_TITLE, call.getString("title", "Connected"))
            .putExtra(ConnectionService.EXTRA_TEXT, call.getString("text", ""));
        try {
            ContextCompat.startForegroundService(getContext(), intent);
            call.resolve();
        } catch (Exception e) {
            // Android doesn't allow starting it while the app is in the background; the app tries again when it's opened.
            call.reject("Couldn't stay connected in the background: " + e.getMessage());
        }
    }

    /** Changes what the ongoing notification says: { title, text }. Works in the background. */
    @PluginMethod
    public void update(PluginCall call) {
        boolean running = ConnectionService.update(getContext(), call.getString("title", ""), call.getString("text", ""));
        JSObject result = new JSObject();
        result.put("running", running);
        call.resolve(result);
    }

    /** { busy }: a task is running on the computer. Works in the background. */
    @PluginMethod
    public void setBusy(PluginCall call) {
        ConnectionService.setBusy(Boolean.TRUE.equals(call.getBoolean("busy", false)));
        call.resolve();
    }

    @PluginMethod
    public void stop(PluginCall call) {
        getContext().stopService(new Intent(getContext(), ConnectionService.class));
        call.resolve();
    }

    /** Shows a notification now: { id, channel: 'waiting' | 'finished', title, body }. */
    @PluginMethod
    public void notify(PluginCall call) {
        String problem = notificationProblem(call.getString("channel", ConnectionService.CHANNEL_FINISHED));
        if (problem != null) {
            call.reject(problem);
            return;
        }
        int id = call.getInt("id", 0);
        String channel = call.getString("channel", ConnectionService.CHANNEL_FINISHED);
        boolean waiting = ConnectionService.CHANNEL_WAITING.equals(channel);
        String body = call.getString("body", "");
        NotificationCompat.Builder b = new NotificationCompat.Builder(getContext(), channel)
            .setSmallIcon(R.drawable.ic_stat_rulz)
            .setColor(0xFFED000D)
            .setContentTitle(call.getString("title", ""))
            .setContentText(body)
            .setStyle(new NotificationCompat.BigTextStyle().bigText(body))
            .setContentIntent(ConnectionService.openApp(getContext(), "project", id))
            .setAutoCancel(true)
            .setPriority(waiting ? NotificationCompat.PRIORITY_HIGH : NotificationCompat.PRIORITY_DEFAULT)
            .setCategory(waiting ? NotificationCompat.CATEGORY_REMINDER : NotificationCompat.CATEGORY_STATUS);
        try {
            NotificationManagerCompat.from(getContext()).notify(id, b.build());
            call.resolve();
        } catch (SecurityException e) {
            call.reject("Android didn't allow the notification: " + e.getMessage());
        }
    }

    /** { id } */
    @PluginMethod
    public void cancel(PluginCall call) {
        NotificationManagerCompat.from(getContext()).cancel(call.getInt("id", 0));
        call.resolve();
    }

    /** { allowed, problem }: whether notifications can show at all, and if not, why. */
    @PluginMethod
    public void notificationStatus(PluginCall call) {
        String problem = notificationProblem(ConnectionService.CHANNEL_WAITING);
        if (problem == null) problem = notificationProblem(ConnectionService.CHANNEL_FINISHED);
        JSObject result = new JSObject();
        result.put("allowed", NotificationManagerCompat.from(getContext()).areNotificationsEnabled());
        result.put("problem", problem);
        call.resolve(result);
    }

    /** Why a notification on this channel wouldn't show, in plain words, or null. */
    private String notificationProblem(String channelId) {
        if (!NotificationManagerCompat.from(getContext()).areNotificationsEnabled()) {
            return "Notifications are turned off for RulZ Agent in Android's settings.";
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = getContext().getSystemService(NotificationManager.class).getNotificationChannel(channelId);
            if (channel != null && channel.getImportance() == NotificationManager.IMPORTANCE_NONE) {
                return "\"" + channel.getName() + "\" notifications are turned off in Android's settings.";
            }
        }
        return null;
    }

    /** Whether Android lets the app run without battery limits. */
    @PluginMethod
    public void batteryStatus(PluginCall call) {
        PowerManager pm = (PowerManager) getContext().getSystemService(android.content.Context.POWER_SERVICE);
        JSObject result = new JSObject();
        result.put("unrestricted", pm.isIgnoringBatteryOptimizations(getContext().getPackageName()));
        call.resolve(result);
    }

    @PluginMethod
    public void openBatterySettings(PluginCall call) {
        open(new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS), call);
    }

    @PluginMethod
    public void openNotificationSettings(PluginCall call) {
        Intent intent;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            intent = new Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).putExtra(Settings.EXTRA_APP_PACKAGE, getContext().getPackageName());
        } else {
            intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:" + getContext().getPackageName()));
        }
        open(intent, call);
    }

    private void open(Intent intent, PluginCall call) {
        try {
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            getContext().startActivity(intent);
            call.resolve();
        } catch (Exception e) {
            call.reject("Couldn't open Android settings: " + e.getMessage());
        }
    }
}
