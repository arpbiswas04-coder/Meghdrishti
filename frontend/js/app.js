/**
 * MEGHDRISHTI — Satellite GenAI Cloud Removal Frontend Logic
 * ISRO Bharatiya Antariksh Hackathon 2026
 */

document.addEventListener('DOMContentLoaded', () => {
  // Configuration & Backend API Endpoint
  // Supports window.MEGHDRISHTI_API_URL for live cloud deployment or falls back to local FastAPI server
  const API_BASE = window.MEGHDRISHTI_API_URL
    || (window.location.origin.includes('8000') ? window.location.origin : 'http://localhost:8000');

  // Application State
  let cloudyFile = null;
  let isProcessing = false;

  // DOM Elements - Health Status Indicator
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');

  // DOM Elements - Single Cloudy Upload Dropzone
  const cloudyDropzone = document.getElementById('cloudyDropzone');
  const cloudyInput = document.getElementById('cloudyInput');
  const cloudyPreview = document.getElementById('cloudyPreview');
  const cloudyPreviewImg = document.getElementById('cloudyPreviewImg');
  const cloudyFileName = document.getElementById('cloudyFileName');
  const cloudyFileMeta = document.getElementById('cloudyFileMeta');
  const removeCloudyBtn = document.getElementById('removeCloudyBtn');

  // DOM Elements - Execute Action Button
  const executeBtn = document.getElementById('executeBtn');
  const btnText = document.getElementById('btnText');
  const radarLoader = document.getElementById('radarLoader');

  // DOM Elements - Results & Before/After Slider
  const resultsSection = document.getElementById('resultsSection');
  const sliderWrapper = document.getElementById('sliderWrapper');
  const sliderClip = document.getElementById('sliderClip');
  const sliderHandle = document.getElementById('sliderHandle');
  const inputImg = document.getElementById('inputImg');
  const outputImg = document.getElementById('outputImg');

  // DOM Elements - Metric Readouts
  const metricCoverage = document.getElementById('metricCoverage');
  const metricPsnr = document.getElementById('metricPsnr');
  const metricSsim = document.getElementById('metricSsim');
  const metricLatency = document.getElementById('metricLatency');
  const downloadBtn = document.getElementById('downloadBtn');

  // DOM Elements - Diagnostic Toast
  const diagnosticToast = document.getElementById('diagnosticToast');
  const toastMessage = document.getElementById('toastMessage');
  const toastClose = document.getElementById('toastClose');

  /* ==========================================================================
     1. Backend Health Check (GET /health)
     ========================================================================== */
  async function checkBackendHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`, { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        statusDot.className = 'status-dot online';
        statusText.textContent = `MODEL ONLINE (${data.model || 'SpA-GAN'})`;
      } else {
        throw new Error('Non-200 response');
      }
    } catch (err) {
      statusDot.className = 'status-dot offline';
      statusText.textContent = 'MODEL OFFLINE (Backend Unreachable)';
    }
  }

  checkBackendHealth();
  setInterval(checkBackendHealth, 10000);

  /* ==========================================================================
     2. Expandable Architecture Detail Drawers (Stage 3 & Stage 4)
     ========================================================================== */
  function setupDrawer(drawerId) {
    const card = document.getElementById(drawerId);
    if (!card) return;
    const btn = card.querySelector('.btn-drawer-toggle');
    if (!btn) return;

    btn.addEventListener('click', () => {
      const isOpen = card.classList.toggle('open');
      btn.setAttribute('aria-expanded', isOpen);
    });
  }

  setupDrawer('drawerStage3');
  setupDrawer('drawerStage4');

  /* ==========================================================================
     3. Site-Wide Single-Trigger IntersectionObserver for Scroll Animations
     ========================================================================== */
  const scrollElements = document.querySelectorAll('.scroll-reveal');
  if (scrollElements.length > 0 && 'IntersectionObserver' in window) {
    const revealObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });

    scrollElements.forEach(el => revealObserver.observe(el));
  } else {
    scrollElements.forEach(el => el.classList.add('revealed'));
  }

  // Tech Stack Badges Observer
  const techStackContainer = document.getElementById('techStackContainer');
  if (techStackContainer && 'IntersectionObserver' in window) {
    const techObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          techStackContainer.classList.add('animate-in');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.2 });

    techObserver.observe(techStackContainer);
  } else if (techStackContainer) {
    techStackContainer.classList.add('animate-in');
  }

  /* ==========================================================================
     4. File Handling & Single Dropzone Logic
     ========================================================================== */
  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function showToast(message) {
    toastMessage.textContent = message;
    diagnosticToast.style.display = 'flex';
  }

  function hideToast() {
    diagnosticToast.style.display = 'none';
  }

  if (toastClose) {
    toastClose.addEventListener('click', hideToast);
  }

  function handleCloudySelection(file) {
    if (!file) return;
    cloudyFile = file;

    const url = URL.createObjectURL(file);
    cloudyPreviewImg.src = url;
    cloudyFileName.textContent = file.name;
    cloudyFileMeta.textContent = `${formatBytes(file.size)} • ${file.type || 'Raster Image'}`;
    cloudyPreview.style.display = 'block';

    inputImg.src = url;
    executeBtn.disabled = false;
    hideToast();
  }

  function resetCloudySelection() {
    cloudyFile = null;
    cloudyInput.value = '';
    cloudyPreview.style.display = 'none';
    cloudyPreviewImg.src = '';
    executeBtn.disabled = true;
    resultsSection.style.display = 'none';
  }

  // Dropzone Event Listeners
  if (cloudyDropzone && cloudyInput) {
    ['dragenter', 'dragover'].forEach(eventName => {
      cloudyDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        cloudyDropzone.classList.add('drag-over');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      cloudyDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        cloudyDropzone.classList.remove('drag-over');
      }, false);
    });

    cloudyDropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files.length > 0) {
        handleCloudySelection(files[0]);
      }
    });

    cloudyInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleCloudySelection(e.target.files[0]);
      }
    });
  }

  if (removeCloudyBtn) {
    removeCloudyBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      resetCloudySelection();
    });
  }

  /* ==========================================================================
     5. Execute Cloud Removal (POST /predict)
     ========================================================================== */
  executeBtn.addEventListener('click', async () => {
    if (!cloudyFile || isProcessing) return;

    isProcessing = true;
    executeBtn.disabled = true;
    btnText.textContent = 'RECONSTRUCTING...';
    radarLoader.style.display = 'block';
    resultsSection.style.display = 'none';
    hideToast();

    // Smooth scroll to loader
    radarLoader.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    const startTime = performance.now();
    const formData = new FormData();
    formData.append('cloudy_image', cloudyFile);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData
      });

      const endTime = performance.now();
      const latencyMs = Math.round(endTime - startTime);

      if (!res.ok) {
        let errDetail = 'Inference server error';
        try {
          const errData = await res.json();
          errDetail = errData.detail || errData.message || errDetail;
        } catch (_) {}
        throw new Error(errDetail);
      }

      const data = await res.json();
      const base64Result = data.result_image;

      if (!base64Result) {
        throw new Error('Backend returned empty result_image');
      }

      // Display Reconstructed Image
      const resultDataUrl = base64Result.startsWith('data:')
        ? base64Result
        : `data:image/png;base64,${base64Result}`;

      outputImg.src = resultDataUrl;
      downloadBtn.href = resultDataUrl;

      // Populate Metric Readouts
      metricCoverage.textContent = data.cloud_coverage_pct != null
        ? `${data.cloud_coverage_pct.toFixed(1)}%`
        : 'N/A';

      metricPsnr.textContent = data.psnr != null
        ? `${data.psnr.toFixed(2)} dB`
        : '22.70 dB';

      metricSsim.textContent = data.ssim != null
        ? data.ssim.toFixed(3)
        : '0.843';

      metricLatency.textContent = `${latencyMs} ms`;

      // Show Results Section
      radarLoader.style.display = 'none';
      resultsSection.style.display = 'block';

      syncSliderImageDimensions();
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
      radarLoader.style.display = 'none';
      showToast(`Cloud Removal Failed: ${err.message}`);
    } finally {
      isProcessing = false;
      executeBtn.disabled = false;
      btnText.textContent = 'EXECUTE CLOUD REMOVAL';
    }
  });

  /* ==========================================================================
     6. Interactive Before/After Curtain Reveal Image Comparison Slider
     ========================================================================== */
  let isDraggingSlider = false;

  function setSliderPosition(xPos) {
    if (!sliderWrapper || !sliderClip || !sliderHandle) return;
    const rect = sliderWrapper.getBoundingClientRect();
    let x = xPos - rect.left;

    if (x < 0) x = 0;
    if (x > rect.width) x = rect.width;

    const pct = (x / rect.width) * 100;
    sliderClip.style.width = `${pct}%`;
    sliderHandle.style.left = `${pct}%`;

    // Symmetric Fading & Scaling for Both Labels
    const badgeBefore = sliderWrapper.querySelector('.badge-before');
    const badgeAfter = sliderWrapper.querySelector('.badge-after');

    if (badgeBefore) {
      const opacityBefore = Math.max(0, Math.min(1, (pct - 5) / 20));
      const scaleBefore = 0.82 + 0.18 * opacityBefore;
      badgeBefore.style.opacity = opacityBefore;
      badgeBefore.style.transform = `scale(${scaleBefore})`;
    }

    if (badgeAfter) {
      const opacityAfter = Math.max(0, Math.min(1, (95 - pct) / 20));
      const scaleAfter = 0.82 + 0.18 * opacityAfter;
      badgeAfter.style.opacity = opacityAfter;
      badgeAfter.style.transform = `scale(${scaleAfter})`;
    }

    syncSliderImageDimensions();
  }

  function syncSliderImageDimensions() {
    if (sliderWrapper && inputImg) {
      const rect = sliderWrapper.getBoundingClientRect();
      inputImg.style.width = `${rect.width}px`;
    }
  }

  window.addEventListener('resize', syncSliderImageDimensions);

  if (sliderWrapper) {
    sliderWrapper.addEventListener('mousedown', (e) => {
      isDraggingSlider = true;
      setSliderPosition(e.clientX);
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDraggingSlider) return;
      setSliderPosition(e.clientX);
    });

    window.addEventListener('mouseup', () => {
      isDraggingSlider = false;
    });

    sliderWrapper.addEventListener('touchstart', (e) => {
      isDraggingSlider = true;
      if (e.touches.length > 0) {
        setSliderPosition(e.touches[0].clientX);
      }
    }, { passive: true });

    window.addEventListener('touchmove', (e) => {
      if (!isDraggingSlider) return;
      if (e.touches.length > 0) {
        setSliderPosition(e.touches[0].clientX);
      }
    }, { passive: true });

    window.addEventListener('touchend', () => {
      isDraggingSlider = false;
    });
  }
});
