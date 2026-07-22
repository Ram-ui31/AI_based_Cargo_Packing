/* ARGO 3D packing viewer -- shared by the "Get a demo" and live-result pages.
   Renders every ULD as a wireframe box, laid out side by side along X, with
   each placed package drawn as a solid box colored by Priority/Economy.
   Package coordinate axes (x=Length, y=Width, z=Height) are remapped so
   three.js's Y (up) corresponds to the real-world Height axis. */

const ArgoViewer = (function () {
  const PRIORITY_COLOR = 0xe8b23a;
  const ECONOMY_COLOR = 0x5f8fb0;
  const ULD_LINE_COLOR = 0xcbb490;
  const GAP = 40;

  function mount(container) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050403);

    const camera = new THREE.PerspectiveCamera(
      45, container.clientWidth / container.clientHeight, 1, 5000
    );
    camera.position.set(300, 260, 500);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const dir = new THREE.DirectionalLight(0xfff2d8, 0.9);
    dir.position.set(400, 600, 300);
    scene.add(dir);
    const dir2 = new THREE.DirectionalLight(0x6f8fb0, 0.35);
    dir2.position.set(-300, 200, -400);
    scene.add(dir2);

    const group = new THREE.Group();
    scene.add(group);

    let raf = null;
    function animate() {
      raf = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    function onResize() {
      const w = container.clientWidth, h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    window.addEventListener('resize', onResize);

    let uldOffsets = {};   // ULD_ID -> {offsetX, cx, cz} for camera focusing
    let uldGroups = {};    // ULD_ID -> THREE.Group (for show/hide filtering)
    let lastData = null;
    let selectMesh = null;    // currently highlighted package mesh
    let selectCb = null;      // user-registered onPackageSelect callback

    function clear() {
      while (group.children.length) group.remove(group.children[0]);
      uldOffsets = {};
      uldGroups = {};
      selectMesh = null;
      if (selectCb) selectCb(null);
    }

    function addBoxOutline(parent, w, h, d, color) {
      const geo = new THREE.BoxGeometry(w, h, d);
      const edges = new THREE.EdgesGeometry(geo);
      const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color, linewidth: 1.5 }));
      parent.add(line);
      return line;
    }

    function renderData(data) {
      clear();
      lastData = data;
      const { ulds, packages, placements } = data;
      const pkgById = {};
      (packages || []).forEach((p) => { pkgById[p.Package_ID] = p; });

      let offsetX = 0;
      const sortedUlds = [...ulds];
      sortedUlds.forEach((uld) => {
        const L = uld.Length, W = uld.Width, H = uld.Height;
        const uldGroup = new THREE.Group();
        uldGroup.position.set(offsetX + L / 2, H / 2, 0);
        group.add(uldGroup);

        // wireframe ULD shell
        addBoxOutline(uldGroup, L, H, W, ULD_LINE_COLOR);

        // floor plate for visual grounding
        const floorGeo = new THREE.PlaneGeometry(L, W);
        const floorMat = new THREE.MeshStandardMaterial({
          color: 0x151005, side: THREE.DoubleSide, transparent: true, opacity: 0.55,
        });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -H / 2 + 0.2;
        uldGroup.add(floor);

        // ULD label sprite
        uldGroup.add(makeLabel(uld.ULD_ID, L, H));

        uldOffsets[uld.ULD_ID] = { offsetX, L, W, H };
        uldGroups[uld.ULD_ID] = uldGroup;
        offsetX += L + GAP;
      });

      (placements || []).forEach((p) => {
        if (p.reason !== 'placed' || p.ULD_ID === 'NONE') return;
        const off = uldOffsets[p.ULD_ID];
        if (!off) return;
        const uldGroup = uldGroups[p.ULD_ID];
        const w = p.x1 - p.x0, h = p.z1 - p.z0, d = p.y1 - p.y0;
        if (w <= 0 || h <= 0 || d <= 0) return;

        const pkg = pkgById[p.Package_ID];
        const isPriority = pkg && String(pkg.Type).trim().toLowerCase() === 'priority';
        const color = isPriority ? PRIORITY_COLOR : ECONOMY_COLOR;

        const geo = new THREE.BoxGeometry(w * 0.96, h * 0.96, d * 0.96);
        const mat = new THREE.MeshStandardMaterial({
          color, roughness: 0.55, metalness: 0.08, transparent: true, opacity: 0.92,
        });
        const mesh = new THREE.Mesh(geo, mat);
        // local position within this ULD's own group (offset already applied via group.position)
        mesh.position.set(
          (p.x0 + p.x1) / 2 - off.L / 2,
          (p.z0 + p.z1) / 2 - off.H / 2,
          (p.y0 + p.y1) / 2 - off.W / 2
        );
        mesh.userData.pkgInfo = {
          id: p.Package_ID,
          uld: p.ULD_ID,
          type: pkg ? pkg.Type : '—',
          weight: pkg ? pkg.Weight : null,
          x0: p.x0, y0: p.y0, z0: p.z0,
          x1: p.x1, y1: p.y1, z1: p.z1,
        };
        mesh.userData.baseColor = color;
        uldGroup.add(mesh);

        const edges = new THREE.EdgesGeometry(geo);
        const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x000000, opacity: 0.25, transparent: true }));
        mesh.add(line);
      });

      focusAll();
    }

    function makeLabel(text, uldL, uldH) {
      const canvas = document.createElement('canvas');
      canvas.width = 256; canvas.height = 64;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'rgba(0,0,0,0)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.font = 'bold 40px sans-serif';
      ctx.fillStyle = '#e8c96a';
      ctx.textAlign = 'center';
      ctx.fillText(text, 128, 46);
      const tex = new THREE.CanvasTexture(canvas);
      const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
      const sprite = new THREE.Sprite(mat);
      sprite.scale.set(60, 15, 1);
      sprite.position.set(0, uldH / 2 + 18, 0);
      return sprite;
    }

    function focusAll() {
      const ids = Object.keys(uldOffsets);
      if (!ids.length) return;
      const totalWidth = ids.reduce((acc, id) => acc + uldOffsets[id].L + GAP, 0);
      controls.target.set(totalWidth / 2 - GAP / 2, 60, 0);
      camera.position.set(totalWidth / 2, 220, Math.max(420, totalWidth * 0.55));
      controls.update();
      Object.values(uldGroups).forEach((g) => (g.visible = true));
    }

    function focusUld(uldId) {
      if (uldId === 'all' || !uldOffsets[uldId]) { focusAll(); return; }
      Object.entries(uldGroups).forEach(([id, g]) => { g.visible = id === uldId; });
      const off = uldOffsets[uldId];
      controls.target.set(off.offsetX + off.L / 2, off.H / 2, 0);
      camera.position.set(off.offsetX + off.L / 2 + off.L * 0.9, off.H * 1.1, off.W * 1.6);
      controls.update();
    }

    function getUldStats(uldId) {
      if (!lastData || !uldOffsets[uldId]) return null;
      const uld = lastData.ulds.find((u) => u.ULD_ID === uldId);
      const pkgById = {};
      (lastData.packages || []).forEach((p) => { pkgById[p.Package_ID] = p; });
      let weightUsed = 0;
      let volumeUsed = 0;
      (lastData.placements || []).forEach((p) => {
        if (p.reason !== 'placed' || p.ULD_ID !== uldId) return;
        const pkg = pkgById[p.Package_ID];
        if (pkg) weightUsed += pkg.Weight;
        volumeUsed += (p.x1 - p.x0) * (p.y1 - p.y0) * (p.z1 - p.z0);
      });
      const weightLimit = uld.Weight_Limit;
      const volumeCapacity = uld.Length * uld.Width * uld.Height;
      return {
        uldId,
        weightUsed, weightLimit,
        weightPct: weightLimit ? (100 * weightUsed / weightLimit) : 0,
        volumeUsed: volumeUsed / 1e6, volumeCapacity: volumeCapacity / 1e6, // cm^3 -> m^3
        volumePct: volumeCapacity ? (100 * volumeUsed / volumeCapacity) : 0,
      };
    }

    function onPackageSelect(cb) {
      selectCb = cb;
    }

    function setSelected(mesh) {
      if (selectMesh) {
        selectMesh.material.emissive && selectMesh.material.emissive.setHex(0x000000);
      }
      selectMesh = mesh;
      if (selectMesh) {
        selectMesh.material.emissive.setHex(0x333333);
      }
    }

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let downPos = null;

    function toNDC(evt) {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((evt.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((evt.clientY - rect.top) / rect.height) * 2 + 1;
    }

    renderer.domElement.addEventListener('pointerdown', (evt) => {
      downPos = { x: evt.clientX, y: evt.clientY };
    });

    renderer.domElement.addEventListener('pointerup', (evt) => {
      if (!downPos) return;
      const moved = Math.hypot(evt.clientX - downPos.x, evt.clientY - downPos.y);
      downPos = null;
      if (moved > 5) return; // was a drag/orbit, not a click
      toNDC(evt);
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(group.children, true);
      const hit = hits.find((h) => h.object.userData && h.object.userData.pkgInfo);
      if (hit) {
        setSelected(hit.object);
        if (selectCb) selectCb(hit.object.userData.pkgInfo);
      } else {
        setSelected(null);
        if (selectCb) selectCb(null);
      }
    });

    function dispose() {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      controls.dispose();
      renderer.dispose();
    }

    return { renderData, focusUld, getUldStats, onPackageSelect, dispose };
  }

  return { mount };
})();
