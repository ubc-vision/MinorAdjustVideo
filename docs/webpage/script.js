const slider = document.getElementById('slider');
const sliderImage = document.getElementById('slider-image');
const sliderVideo = document.getElementById('slider-video');
const btnHighRes = document.getElementById('btn-highres');
const btnLowRes = document.getElementById('btn-lowres');

if (!slider) throw new Error('Missing element: #slider');
if (!sliderImage) throw new Error('Missing element: #slider-image');
if (!sliderVideo) throw new Error('Missing element: #slider-video');
if (!btnHighRes) throw new Error('Missing element: #btn-highres');
if (!btnLowRes) throw new Error('Missing element: #btn-lowres');

const frameFolderHighRes = "resources/images/optim/traj_evolution2/high_res/maps/";
const frameFolderLowRes = "resources/images/optim/traj_evolution2/low_res/maps/";
const videoSrc = "resources/images/optim/traj_evolution2/video/video_disp.0000.mp4";

// Folder currently contains a full grid: t=0000..0004 and opt_idx=0000..0004
const tCount = 5;
const optCount = 5;

function pad4(n) {
  return String(n).padStart(4, "0");
}

let currentMapRes = "high"; // "high" | "low"

function frameFolder() {
  if (currentMapRes === "high") return frameFolderHighRes;
  if (currentMapRes === "low") return frameFolderLowRes;
  throw new Error(`Unknown map resolution: ${currentMapRes}`);
}

function frameFilename(t, opt) {
  const prefix = currentMapRes === "high" ? "up.2.2.0" : "down.0.0.0";
  return `${prefix}_maps_diff_map-attn_map_spatial_t_${pad4(t)}_opt_idx_${pad4(opt)}.png`;
}

// Build frames in requested order:
// t=0000 opt=0000..0004, then t=0001 opt=0000..0004, ...
const frames = [];
for (let t = 0; t < tCount; t++) {
  for (let opt = 0; opt < optCount; opt++) {
    frames.push({ t, opt });
}
}

slider.min = 0;
slider.max = frames.length - 1;
slider.value = 0;

let autoplayInterval = null;
let currentIndex = 0;
let mode = "frames"; // "frames" | "video"
let videoPlaysSoFar = 0;
const videoPlaysPerCycle = 3;

function showFramesMode() {
  mode = "frames";
  sliderVideo.pause();
  sliderVideo.classList.remove("media-visible");
  sliderVideo.classList.add("media-hidden");
}

function showVideoMode() {
  mode = "video";
  sliderVideo.classList.remove("media-hidden");
  sliderVideo.classList.add("media-visible");
}

function setFrame(index) {
  if (index < 0 || index >= frames.length) {
    throw new Error(`Frame index out of range: ${index} (frames=${frames.length})`);
  }
  currentIndex = index;
  slider.value = String(index);
  sliderImage.src = frameFolder() + frameFilename(frames[index].t, frames[index].opt);
}

function stopAutoplay() {
  if (autoplayInterval !== null) {
    clearInterval(autoplayInterval);
    autoplayInterval = null;
  }
}

function playEndVideoOnce() {
  stopAutoplay();
  showVideoMode();
  videoPlaysSoFar = 0;

  // Ensure correct src even if <source> changes later.
  if (sliderVideo.currentSrc && !sliderVideo.currentSrc.endsWith(videoSrc)) {
    sliderVideo.src = videoSrc;
  } else if (!sliderVideo.currentSrc) {
    sliderVideo.src = videoSrc;
  }

  sliderVideo.currentTime = 0;

  const p = sliderVideo.play();
  if (p && typeof p.catch === "function") {
    p.catch((err) => {
      console.error("Video play() failed:", err);
      throw err;
    });
  }
}

function startAutoplay() {
  stopAutoplay();
  if (mode !== "frames") return;

  autoplayInterval = setInterval(() => {
    if (currentIndex === frames.length - 1) {
      playEndVideoOnce();
      return;
    }
    setFrame(currentIndex + 1);
  }, 100); // keep current speed
}

// Cycle: video plays N times, then go back to sliding from start.
sliderVideo.addEventListener("ended", () => {
  videoPlaysSoFar += 1;

  if (videoPlaysSoFar < videoPlaysPerCycle) {
    sliderVideo.currentTime = 0;
    const p = sliderVideo.play();
    if (p && typeof p.catch === "function") {
      p.catch((err) => {
        console.error("Video replay play() failed:", err);
        throw err;
      });
}
    return;
  }

  setTimeout(() => {
    showFramesMode();
    setFrame(0);
startAutoplay();
  }, 1000);
});

// Pause/resume on hover (frames mode only)
slider.addEventListener('mouseenter', () => {
  if (mode === "frames") stopAutoplay();
});
slider.addEventListener('mouseleave', () => {
  if (mode === "frames") startAutoplay();
});

// Manual slider control:
// If user drags to last frame, immediately play the video.
slider.addEventListener('input', () => {
  showFramesMode();
  const idx = parseInt(slider.value, 10);
  setFrame(idx);
  if (idx === frames.length - 1) {
    playEndVideoOnce();
  }
});

function setMapRes(newRes) {
  if (newRes !== "high" && newRes !== "low") {
    throw new Error(`Invalid map resolution: ${newRes}`);
  }

  currentMapRes = newRes;
  btnHighRes.classList.toggle("is-active", newRes === "high");
  btnLowRes.classList.toggle("is-active", newRes === "low");

  // Reset to frame 0 on toggle (per your spec)
  showFramesMode();
  stopAutoplay();
  setFrame(0);
  startAutoplay();
}

btnHighRes.addEventListener("click", () => setMapRes("high"));
btnLowRes.addEventListener("click", () => setMapRes("low"));

// Initialize
showFramesMode();
setFrame(0);
startAutoplay();