# Wattwise media credits

The login and dashboard use local media. The factory footage and industrial photograph illustrate the application domain. The project’s experimental video and numerical results come from the existing Defence 2026 evidence.

| Local asset | Creator and source | Licence and edits |
| --- | --- | --- |
| `media/robotics/factory-motion.mp4` | Usman AbdulrasheedGambo, [Industrial Robot Arm in High-Tech Factory](https://www.pexels.com/video/industrial-robot-arm-in-high-tech-factory-32386532/) | [Pexels License](https://www.pexels.com/license/). Trimmed, crossfaded at the loop, resized to 1440 × 810 at 25 fps, and compressed as silent H.264. |
| `media/robotics/factory-poster.jpg`, `media/robotics/factory-poster-mobile.jpg` | Frames from the same licensed footage | [Pexels License](https://www.pexels.com/license/). Optimized progressive JPEGs with a tighter mobile crop. |
| `media/robotics/robot-arm.jpg` | Freek Wolsink, [Industrial Robot Arm in a Manufacturing Facility](https://www.pexels.com/photo/industrial-robot-arm-in-a-manufacturing-facility-34207359/) | [Pexels License](https://www.pexels.com/license/). Resized and optimized as progressive JPEG. |
| `media/evidence-posters/*.jpg` | Frames at approximately 0.4 seconds from the corresponding project recordings in `DEFENCE_2026/videos/` | Project evidence thumbnails resized and optimized as progressive JPEG. The original recordings and measurements remain in the defence folder. |

## Replace decorative assets

Put replacement files under `media/robotics/` and update the paths in [robotics-media.js](robotics-media.js). Update the visible credits in the login and dashboard and record the new source and terms here. Leave `loginVideo` empty for a still-only login. The default login uses a smaller poster on mobile and displays video on wider screens when motion is allowed. The background can also be paused from the login header.

To override the dashboard reference photo, save your own image as `robot-wireframe.png` in this assets directory. **Choose robot image** on the login page previews an image and passes the selection to the dashboard for the current tab session.
