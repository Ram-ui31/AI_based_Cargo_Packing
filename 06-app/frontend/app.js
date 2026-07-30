/* ARGO frontend -- single-page app: navigation between sections, the
   "Get a demo" precomputed viewer, "Run your own CSV" (a live backend call,
   for judges/visitors who don't want to download the desktop app), and
   "Use our product" which points visitors at the downloadable desktop app. */

// Base URL for the live-pack API (/api/pack, /api/status). Empty string
// means same-origin, which is correct when this frontend is served BY that
// same backend (the desktop app's bundled server, or the backend deployed
// on its own e.g. a Hugging Face Space). When this frontend is instead
// served statically from a different origin (e.g. GitHub Pages), point this
// at the deployed backend's absolute URL so the cross-origin fetch calls
// below resolve correctly (the backend enables CORS for this).
const ARGO_API_BASE = window.ARGO_API_BASE || '';

(function () {
  const pages = document.querySelectorAll('.page');
  function showPage(id) {
    pages.forEach((p) => p.classList.toggle('active', p.id === id));
  }

  document.querySelectorAll('[data-goto]').forEach((el) => {
    el.addEventListener('click', () => showPage(el.getAttribute('data-goto')));
  });

  document.getElementById('hs-about').addEventListener('click', () => showPage('page-about'));
  document.getElementById('hs-demo').addEventListener('click', () => {
    showPage('page-demo-viz');
    setMode('sample');
  });
  document.getElementById('hs-product').addEventListener('click', () => showPage('page-download'));
  document.getElementById('btn-try-live').addEventListener('click', () => {
    showPage('page-demo-viz');
    setMode('upload');
  });

  // ---------------------------------------------------------------------
  // Use Our Product -- highlight the download card matching the visitor's OS
  // ---------------------------------------------------------------------
  (function highlightMatchingOS() {
    const ua = navigator.userAgent || '';
    const platform = navigator.platform || '';
    let match = null;
    if (/Mac/i.test(platform) || /Macintosh/i.test(ua)) match = 'dl-mac';
    else if (/Win/i.test(platform) || /Windows/i.test(ua)) match = 'dl-windows';
    else if (/Linux/i.test(platform) || /Linux/i.test(ua)) match = 'dl-linux';
    if (match) {
      const card = document.getElementById(match);
      if (card) card.classList.add('recommended');
    }
  })();

  // ---------------------------------------------------------------------
  // About Our Models -> model detail
  // ---------------------------------------------------------------------
  const MODEL_INFO = {
    halley: {
      title: 'Halley',
      subtitle: 'GNN-ranked economy selection',
      body: 'Halley uses the same core ARGO engine as its siblings to place Priority packages, '
        + 'but adds a smart "scout": a small neural network that looks at all the Economy '
        + 'packages together and works out the cleverest order to pack them in, instead of '
        + 'following one fixed rule of thumb.',
      points: [
        'PackageSetRanker: GNN scoring over package + ULD features',
        'Generalizes across differently-sized instances (no fixed normalization)',
        'Best suited when Economy package mix is highly heterogeneous',
      ],
    },
    cherry: {
      title: 'Cherry',
      subtitle: 'Our best-performing model',
      body: 'Cherry starts from the same core engine as its siblings, then goes one step '
        + 'further: it keeps trying to pull a package back out, tidy up the leftover space, '
        + 'and pack things back in more tightly — keeping every change that actually helps, '
        + 'until it can\'t find any more improvements.',
      points: [
        'Centrifuge-evict-refine local search on top of the shared ensemble',
        'Lowest total cost across our benchmark suite',
        'Trades extra compute time for the tightest packing',
      ],
    },
    eclipse: {
      title: 'Eclipse',
      subtitle: 'The clean, fast baseline',
      body: 'Eclipse is the clean, no-frills version of the ARGO engine: it places the '
        + 'Priority packages first, then packs Economy packages in order of how much value '
        + 'each one packs in for its size — no extra fine-tuning pass, just a fast, solid pack.',
      points: [
        'Same clusterer + ensemble placement core as Cherry and Halley',
        'No centrifuge refinement, no GNN ranker — fastest of the three',
        'A strong, dependable middle ground',
      ],
    },
  };

  function openDetail(key) {
    const info = MODEL_INFO[key];
    document.getElementById('detail-title').textContent = info.title;
    document.getElementById('detail-subtitle').textContent = info.subtitle;
    document.getElementById('detail-body').textContent = info.body;
    const list = document.getElementById('detail-list');
    list.innerHTML = '';
    info.points.forEach((pt) => {
      const li = document.createElement('li');
      li.textContent = pt;
      list.appendChild(li);
    });
    showPage('page-detail');
  }

  document.getElementById('hs-halley').addEventListener('click', () => openDetail('halley'));
  document.getElementById('hs-cherry').addEventListener('click', () => openDetail('cherry'));
  document.getElementById('hs-eclipse').addEventListener('click', () => openDetail('eclipse'));

  // ---------------------------------------------------------------------
  // Get a Demo -- precomputed results
  // ---------------------------------------------------------------------
  let demoViewer = null;
  let demoData = null;

  // The photo background (argo_back.png) is meant for the "nothing rendered
  // yet" shell -- upload form, loading, errors. Once an actual packing
  // result is on screen, switch to a plain dark background (like the
  // desktop app) so the boxes/labels aren't competing with a busy photo.
  function setHasResult(hasResult) {
    document.getElementById('page-demo-viz').classList.toggle('has-result', hasResult);
    document.getElementById('demo-side-panel').style.display = hasResult ? '' : 'none';
  }

  // Plain static JSON files (demo_data/), no backend required -- works
  // identically whether served by backend/app.py, the desktop app's bundled
  // server, or a fully static host (GitHub Pages, Cloudflare Pages, etc).
  async function loadDemo(model) {
    const panel = document.getElementById('demo-side-panel');
    setHasResult(false);
    panel.innerHTML = '<h3>Loading…</h3>';
    try {
      const [metrics, placements, ulds, packages] = await Promise.all([
        fetch(`demo_data/${model}/final_metrics.json`).then((r) => { if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); }),
        fetch(`demo_data/${model}/final_placements.json`).then((r) => { if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); }),
        fetch('demo_data/ulds.json').then((r) => { if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); }),
        fetch('demo_data/packages.json').then((r) => { if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); }),
      ]);
      demoData = { metrics, placements, ulds, packages };
    } catch (err) {
      setHasResult(true); // show the panel so the error is actually visible
      panel.innerHTML = `<h3>Error</h3><p>${err.message}</p>`;
      return;
    }
    if (!demoViewer) {
      demoViewer = ArgoViewer.mount(document.getElementById('demo-canvas-wrap'));
      demoViewer.onPackageSelect((info) => showPackageInfoBox('demo-canvas-wrap', info));
    }
    demoViewer.renderData(demoData);
    renderSidePanel(panel, demoData, model, demoViewer);
    setHasResult(true);
  }

  document.getElementById('demo-model-select').addEventListener('change', (e) => {
    loadDemo(e.target.value);
  });

  // ---------------------------------------------------------------------
  // Run your own CSV -- live call to the deployed backend's /api/pack,
  // polling /api/status until done, then reusing the same viewer + side
  // panel as the precomputed demo (identical result shape).
  // ---------------------------------------------------------------------
  const MAX_UPLOAD_BYTES = 2 * 1024 * 1024; // 2 MB
  let uploadFile = null;
  let uploadPollTimer = null;

  function setMode(mode) {
    document.getElementById('mode-tab-sample').classList.toggle('active', mode === 'sample');
    document.getElementById('mode-tab-upload').classList.toggle('active', mode === 'upload');
    document.getElementById('demo-model-group').style.display = mode === 'sample' ? '' : 'none';
    document.getElementById('upload-panel').style.display = mode === 'upload' ? '' : 'none';
    document.getElementById('demo-brand').textContent = mode === 'sample'
      ? 'ARGO · Demo (400-package benchmark instance)'
      : 'ARGO · Run your own CSV';
    if (mode === 'sample') {
      if (uploadPollTimer) { clearInterval(uploadPollTimer); uploadPollTimer = null; }
      loadDemo(document.getElementById('demo-model-select').value);
    } else {
      setHasResult(false); // no result yet in upload mode until a run finishes
      document.getElementById('demo-side-panel').innerHTML = '';
      if (demoViewer) demoViewer.dispose();
      demoViewer = null;
      // dispose() frees GL resources but doesn't remove the canvas itself --
      // without this the last-rendered frame stays frozen on screen behind
      // the upload form.
      document.getElementById('demo-canvas-wrap').innerHTML = '';
    }
  }
  document.getElementById('mode-tab-sample').addEventListener('click', () => setMode('sample'));
  document.getElementById('mode-tab-upload').addEventListener('click', () => setMode('upload'));

  const fileInput = document.getElementById('upload-file-input');
  const runBtn = document.getElementById('upload-run-btn');
  fileInput.addEventListener('change', () => {
    const f = fileInput.files[0];
    if (f && f.size > MAX_UPLOAD_BYTES) {
      alert(`File too large (${(f.size / 1024 / 1024).toFixed(1)} MB) -- max is 2 MB.`);
      fileInput.value = '';
      uploadFile = null;
      runBtn.disabled = true;
      return;
    }
    uploadFile = f || null;
    runBtn.disabled = !uploadFile;
  });

  function setUploadProgress(pct, message) {
    const box = document.getElementById('upload-progress');
    box.style.display = '';
    document.getElementById('upload-progress-fill').style.width = `${pct}%`;
    document.getElementById('upload-progress-msg').textContent = message || '';
  }

  runBtn.addEventListener('click', async () => {
    if (!uploadFile) return;
    runBtn.disabled = true;
    fileInput.disabled = true;
    setUploadProgress(2, 'Uploading...');
    const model = document.getElementById('upload-model-select').value;
    const form = new FormData();
    form.append('model', model);
    form.append('file', uploadFile);
    let jobId;
    try {
      const res = await fetch(`${ARGO_API_BASE}/api/pack`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      ({ job_id: jobId } = await res.json());
    } catch (err) {
      setUploadProgress(0, `Error: ${err.message}`);
      runBtn.disabled = false;
      fileInput.disabled = false;
      return;
    }

    uploadPollTimer = setInterval(async () => {
      let job;
      try {
        const res = await fetch(`${ARGO_API_BASE}/api/status/${jobId}`);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        job = await res.json();
      } catch (err) {
        clearInterval(uploadPollTimer);
        uploadPollTimer = null;
        setUploadProgress(0, `Error: ${err.message}`);
        runBtn.disabled = false;
        fileInput.disabled = false;
        return;
      }
      setUploadProgress(job.progress || 0, job.message || '');
      if (job.status === 'done') {
        clearInterval(uploadPollTimer);
        uploadPollTimer = null;
        runBtn.disabled = false;
        fileInput.disabled = false;
        const panel = document.getElementById('demo-side-panel');
        if (!demoViewer) {
          demoViewer = ArgoViewer.mount(document.getElementById('demo-canvas-wrap'));
          demoViewer.onPackageSelect((info) => showPackageInfoBox('demo-canvas-wrap', info));
        }
        demoViewer.renderData(job.result);
        renderSidePanel(panel, job.result, model, demoViewer);
        setHasResult(true);
      } else if (job.status === 'error') {
        clearInterval(uploadPollTimer);
        uploadPollTimer = null;
        runBtn.disabled = false;
        fileInput.disabled = false;
      }
    }, 1500);
  });

  // ---------------------------------------------------------------------
  // Shared side-panel renderer (metrics + ULD tabs + legend + download)
  // ---------------------------------------------------------------------
  function renderSidePanel(panel, data, modelLabel, viewer) {
    const m = data.metrics;
    panel.innerHTML = '';

    const h3 = document.createElement('h3');
    h3.textContent = (m.model || modelLabel || '').toString();
    panel.appendChild(h3);

    const totalPackages = m.n_priority_total + m.n_economy_total;
    const totalPlaced = (m.n_priority_total - m.n_priority_unplaced) + (m.n_economy_total - m.n_economy_unplaced);
    const rows = [
      ['Total cost', m.total_cost],
      ['Delay cost', m.delay_cost],
      ['Spread cost', m.spread_cost],
      ['Total placed', `${totalPlaced} / ${totalPackages}`],
      ['Priority placed', `${m.n_priority_total - m.n_priority_unplaced} / ${m.n_priority_total}`],
      ['Economy placed', `${m.n_economy_total - m.n_economy_unplaced} / ${m.n_economy_total}`],
      ['Priority ULDs used', m.n_priority_ulds],
      ['Wall time', m.elapsed_seconds ? `${m.elapsed_seconds.toFixed(1)}s` : '—'],
    ];
    rows.forEach(([label, val]) => {
      const row = document.createElement('div');
      row.className = 'metric-row';
      row.innerHTML = `<span>${label}</span><span>${val}</span>`;
      panel.appendChild(row);
    });

    const legend = document.createElement('div');
    legend.className = 'legend';
    legend.innerHTML = `
      <span><span class="legend-dot" style="background:#e8b23a"></span>Priority</span>
      <span><span class="legend-dot" style="background:#5f8fb0"></span>Economy</span>`;
    panel.appendChild(legend);

    const tabsLabel = document.createElement('div');
    tabsLabel.className = 'subtitle';
    tabsLabel.style.marginTop = '10px';
    tabsLabel.textContent = 'Focus a ULD';
    panel.appendChild(tabsLabel);

    const tabs = document.createElement('div');
    tabs.className = 'uld-tabs';
    const uldStats = document.createElement('div');
    uldStats.className = 'uld-stats';
    const ids = ['all', ...data.ulds.map((u) => u.ULD_ID)];
    ids.forEach((id) => {
      const btn = document.createElement('button');
      btn.className = 'uld-tab' + (id === 'all' ? ' active' : '');
      btn.textContent = id === 'all' ? 'All' : id;
      btn.addEventListener('click', () => {
        tabs.querySelectorAll('.uld-tab').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        viewer.focusUld(id);
        renderUldStats(uldStats, viewer, id);
      });
      tabs.appendChild(btn);
    });
    panel.appendChild(tabs);
    panel.appendChild(uldStats);

    const dlBtn = document.createElement('button');
    dlBtn.className = 'download-btn';
    dlBtn.textContent = 'Download .json';
    dlBtn.addEventListener('click', () => downloadJson(data, modelLabel));
    panel.appendChild(dlBtn);
  }

  function renderUldStats(container, viewer, uldId) {
    if (uldId === 'all') {
      container.innerHTML = '';
      return;
    }
    const stats = viewer.getUldStats(uldId);
    if (!stats) {
      container.innerHTML = '';
      return;
    }
    container.innerHTML = `
      <div class="uld-stat-title">${stats.uldId} occupancy</div>
      <div class="metric-row"><span>Weight</span><span>${stats.weightPct.toFixed(1)}% (${stats.weightUsed.toFixed(0)} / ${stats.weightLimit.toFixed(0)} kg)</span></div>
      <div class="metric-row"><span>Volume</span><span>${stats.volumePct.toFixed(1)}% (${stats.volumeUsed.toFixed(2)} / ${stats.volumeCapacity.toFixed(2)} m&sup3;)</span></div>
    `;
  }

  // ---------------------------------------------------------------------
  // Package click -> spec info box
  // ---------------------------------------------------------------------
  function showPackageInfoBox(wrapId, info) {
    const wrap = document.getElementById(wrapId);
    let box = wrap.querySelector('.pkg-info-box');
    if (!info) {
      if (box) box.remove();
      return;
    }
    if (!box) {
      box = document.createElement('div');
      box.className = 'pkg-info-box';
      wrap.appendChild(box);
    }
    const volume = ((info.x1 - info.x0) * (info.y1 - info.y0) * (info.z1 - info.z0) / 1e6);
    const isPriority = String(info.type).trim().toLowerCase() === 'priority';
    box.innerHTML = `
      <button class="pkg-info-close">&times;</button>
      <div class="pkg-info-title">${info.id}</div>
      <div class="metric-row"><span>Type</span><span style="color:${isPriority ? '#e8b23a' : '#5f8fb0'}">${info.type}</span></div>
      <div class="metric-row"><span>Weight</span><span>${info.weight != null ? info.weight + ' kg' : '—'}</span></div>
      <div class="metric-row"><span>Volume</span><span>${volume.toFixed(3)} m&sup3;</span></div>
      <div class="metric-row"><span>Min corner (x,y,z)</span><span>${info.x0}, ${info.y0}, ${info.z0}</span></div>
      <div class="metric-row"><span>Max corner (x,y,z)</span><span>${info.x1}, ${info.y1}, ${info.z1}</span></div>
    `;
    box.querySelector('.pkg-info-close').addEventListener('click', () => box.remove());
  }

  function downloadJson(data, modelLabel) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `argo_${modelLabel || 'result'}_packing.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

})();
