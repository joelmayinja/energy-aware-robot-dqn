(() => {
  'use strict';

  // Change these paths and the matching credits when replacing the local media.
  const media = {
    loginPoster: '/assets/media/robotics/factory-poster.jpg',
    loginMobile: '/assets/media/robotics/factory-poster-mobile.jpg',
    loginVideo: '/assets/media/robotics/factory-motion.mp4',
    dashboardPhoto: '/assets/media/robotics/robot-arm.jpg',
    customPhoto: '/assets/robot-wireframe.png',
    evidencePosterRoot: '/assets/media/evidence-posters/',
  };

  window.WattwiseMedia = {
    evidencePoster(file) {
      const recorded = /^(ik|teacher_original|student_original|student_improved|comparison_2x2)_seed140000_20261127_002702\.mp4$/;
      return recorded.test(file) ? media.evidencePosterRoot + file.replace(/\.mp4$/, '.jpg') : '';
    },
  };

  function loadCustomPhoto(image, update) {
    const probe = new Image();
    probe.onload = () => {
      if (image.dataset.custom === 'true' || image.src.startsWith('blob:')) return;
      update();
      image.src = probe.src;
    };
    probe.src = media.customPhoto;
  }

  function setupLoginMedia() {
    const backdrop = document.querySelector('.login-media');
    if (!backdrop) return;

    const poster = backdrop.querySelector('img');
    backdrop.querySelector('source').srcset = media.loginMobile;
    poster.src = media.loginPoster;
    const preview = document.getElementById('robot-reference');
    loadCustomPhoto(preview, () => preview.closest('.robot-stage').classList.add('has-reference', 'is-custom'));
    placeImagePicker();

    if (!media.loginVideo) return;
    const video = backdrop.querySelector('video');
    const toggle = document.getElementById('motion-toggle');
    const canAnimate = matchMedia('(min-width: 901px) and (prefers-reduced-motion: no-preference)');
    let pausedByUser = false;
    let failed = false;
    let leaving = false;

    function synchronize() {
      toggle.hidden = failed || !canAnimate.matches;
      toggle.textContent = pausedByUser ? 'Play background' : 'Pause background';
      if (leaving || document.hidden || !canAnimate.matches || pausedByUser || failed) {
        video.pause();
        return;
      }
      if (!video.getAttribute('src')) {
        video.poster = media.loginPoster;
        video.src = media.loginVideo;
      }
      video.play().catch(error => {
        if (error.name === 'AbortError') return;
        // Autoplay policies can require a click even for a muted video.
        pausedByUser = true;
        toggle.textContent = 'Play background';
      });
    }

    video.addEventListener('playing', () => backdrop.classList.add('is-playing'));
    video.addEventListener('error', () => {
      failed = true;
      backdrop.classList.remove('is-playing');
      synchronize();
    });
    toggle.addEventListener('click', () => {
      pausedByUser = !pausedByUser;
      synchronize();
    });
    canAnimate.addEventListener('change', () => {
      if (!canAnimate.matches) backdrop.classList.remove('is-playing');
      synchronize();
    });
    document.addEventListener('visibilitychange', synchronize);
    window.addEventListener('pagehide', () => { leaving = true; synchronize(); });
    window.addEventListener('pageshow', () => { leaving = false; synchronize(); });
    synchronize();
  }

  function placeImagePicker() {
    const appearance = document.getElementById('light-appearance');
    const picker = document.getElementById('robot-image-file').closest('label');
    const originalParent = picker.parentElement;
    const next = picker.nextElementSibling;
    const panel = appearance.querySelector('.light-appearance-panel');
    const theme = document.documentElement;

    function synchronize() {
      const light = theme.dataset.mode === 'light';
      appearance.hidden = !light;
      if (light) panel.append(picker);
      else if (picker.parentElement !== originalParent) {
        appearance.open = false;
        originalParent.insertBefore(picker, next);
      }
    }

    new MutationObserver(synchronize).observe(theme, { attributeFilter: ['data-mode'] });
    synchronize();
  }

  function setupDashboardMedia() {
    const heroPhoto = document.getElementById('hero-robot-image');
    if (!heroPhoto || heroPhoto.hasAttribute('src')) return;

    heroPhoto.addEventListener('error', () => {
      heroPhoto.closest('.hero').classList.remove('has-real-image');
      heroPhoto.hidden = true;
    });
    heroPhoto.src = media.dashboardPhoto;
    loadCustomPhoto(heroPhoto, () => {
      heroPhoto.dataset.custom = 'true';
      document.getElementById('hero-image-credit').textContent = 'Custom robot image';
    });
  }

  async function showDefenceTeacherResult() {
    const badge = document.getElementById('defence-teacher-result');
    if (!badge) return;

    try {
      const response = await fetch('/api/simulation-evidence');
      if (!response.ok) return;
      const { policies } = await response.json();
      const teacher = policies.find(policy => policy.policy === 'teacher_original');
      if (!teacher?.episodes || teacher.completions === '') return;
      document.getElementById('defence-teacher-count').textContent =
        teacher.completions + '/' + teacher.episodes;
      badge.hidden = false;
    } catch {
      // Keep the optional result hidden if the local evidence is unavailable.
    }
  }

  setupLoginMedia();
  setupDashboardMedia();
  showDefenceTeacherResult();
})();
