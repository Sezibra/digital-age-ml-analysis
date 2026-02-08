const STORAGE_KEY = "trilog-activities-v1";
const CERT_STORAGE_KEY = "trilog-certificates-v1";
const MAP_KEY_STORAGE = "trilog-google-maps-key-v1";

const sampleActivities = [
  {
    id: crypto.randomUUID(),
    date: "2026-01-28",
    category: "running",
    entryType: "competition",
    title: "Winter City 10K",
    duration: 44,
    distance: 10,
    time: "44:21",
    placement: "38 / 610",
    officialLink: "https://example.com/winter-city-10k",
    location: "Munich",
    lat: 48.1351,
    lng: 11.582,
    notes: "Negative split in second half"
  },
  {
    id: crypto.randomUUID(),
    date: "2026-02-01",
    category: "swimming",
    entryType: "training",
    title: "Technique + pull buoy",
    duration: 55,
    distance: 2.2,
    time: "",
    placement: "",
    officialLink: "",
    location: "",
    lat: null,
    lng: null,
    notes: "Focus on catch"
  },
  {
    id: crypto.randomUUID(),
    date: "2026-02-02",
    category: "gym",
    entryType: "training",
    title: "Lower body strength",
    duration: 65,
    distance: 0,
    time: "",
    placement: "",
    officialLink: "",
    location: "",
    lat: null,
    lng: null,
    notes: "Squat 5x5"
  },
  {
    id: crypto.randomUUID(),
    date: "2026-02-04",
    category: "swimming",
    entryType: "competition",
    title: "Open Water 2K",
    duration: 39,
    distance: 2,
    time: "38:47",
    placement: "16 / 140",
    officialLink: "https://example.com/open-water-2k",
    location: "Nice",
    lat: 43.7102,
    lng: 7.262,
    notes: "Strong finish"
  }
];

const sampleCertificates = [
  {
    id: crypto.randomUUID(),
    title: "10K Finisher",
    org: "Winter City Run",
    date: "2026-01-28",
    fileName: "certificate-demo.jpg",
    fileType: "image",
    dataUrl:
      "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='800' height='500'><rect width='100%25' height='100%25' fill='%23f7f3e8'/><rect x='28' y='28' width='744' height='444' fill='none' stroke='%230d7b84' stroke-width='7'/><text x='50%25' y='40%25' dominant-baseline='middle' text-anchor='middle' font-size='48' font-family='Arial' fill='%23202124'>Certificate of Participation</text><text x='50%25' y='58%25' dominant-baseline='middle' text-anchor='middle' font-size='28' font-family='Arial' fill='%2359616b'>Winter City 10K</text></svg>"
  }
];

let activities = loadFromStorage(STORAGE_KEY, sampleActivities);
let certificates = loadFromStorage(CERT_STORAGE_KEY, sampleCertificates);
let volumeChart;
let categoryChart;
let map;
let markersLayer;
let googleMap;
let googleMarkers = [];
let mapMode = "osm";

const activityForm = document.getElementById("activityForm");
const certificateForm = document.getElementById("certificateForm");
const activityList = document.getElementById("activityList");
const certificateGallery = document.getElementById("certificateGallery");
const heroStats = document.getElementById("heroStats");
const routineCards = document.getElementById("routineCards");
const mapSettingsForm = document.getElementById("mapSettingsForm");
const googleMapsKeyInput = document.getElementById("googleMapsKey");
const mapModeHint = document.getElementById("mapModeHint");

init();

async function init() {
  const savedMapKey = localStorage.getItem(MAP_KEY_STORAGE) || "";
  googleMapsKeyInput.value = savedMapKey;
  await initMap(savedMapKey);
  bindEvents();
  render();
}

function bindEvents() {
  activityForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const formData = new FormData(activityForm);

    const activity = {
      id: crypto.randomUUID(),
      date: formData.get("date") || new Date().toISOString().slice(0, 10),
      category: formData.get("category"),
      entryType: formData.get("entryType"),
      title: sanitize(formData.get("title")),
      duration: Number(formData.get("duration") || 0),
      distance: Number(formData.get("distance") || 0),
      time: sanitize(formData.get("time") || ""),
      placement: sanitize(formData.get("placement") || ""),
      officialLink: sanitize(formData.get("officialLink") || ""),
      location: sanitize(formData.get("location") || ""),
      lat: parseCoordinate(formData.get("lat")),
      lng: parseCoordinate(formData.get("lng")),
      notes: sanitize(formData.get("notes") || "")
    };

    activities.push(activity);
    activities = sortByDateDesc(activities);
    saveToStorage(STORAGE_KEY, activities);
    activityForm.reset();
    render();
  });

  certificateForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(certificateForm);
    const file = formData.get("file");

    if (!(file instanceof File) || !file.size) {
      return;
    }

    const dataUrl = await fileToDataUrl(file);
    const certificate = {
      id: crypto.randomUUID(),
      title: sanitize(formData.get("title")),
      org: sanitize(formData.get("org")),
      date: formData.get("date") || new Date().toISOString().slice(0, 10),
      fileName: file.name,
      fileType: file.type.includes("pdf") ? "pdf" : "image",
      dataUrl
    };

    certificates.unshift(certificate);
    saveToStorage(CERT_STORAGE_KEY, certificates);
    certificateForm.reset();
    renderCertificates();
  });

  mapSettingsForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const key = sanitize(googleMapsKeyInput.value);
    localStorage.setItem(MAP_KEY_STORAGE, key);
    await initMap(key);
    renderMapMarkers();
  });
}

function render() {
  renderHeroStats();
  renderRoutineCards();
  renderActivityList();
  renderCharts();
  renderMapMarkers();
  renderCertificates();
}

function renderHeroStats() {
  const allDuration = activities.reduce((sum, item) => sum + (item.duration || 0), 0);
  const allDistance = activities.reduce((sum, item) => sum + (item.distance || 0), 0);
  const streak = calculateStreak(activities);

  const perf = activities
    .filter((a) => a.entryType === "competition")
    .reduce((acc, item) => acc + item.distance * 3 + item.duration * 0.2, 0)
    .toFixed(0);

  heroStats.innerHTML = `
    <div class="quick-stat"><div>Total Duration</div><strong>${Math.round(allDuration)} min</strong></div>
    <div class="quick-stat"><div>Total Distance</div><strong>${allDistance.toFixed(1)} km</strong></div>
    <div class="quick-stat"><div>Consistency Streak</div><strong>${streak} days</strong></div>
    <div class="quick-stat"><div>Performance Index</div><strong>${perf}</strong></div>
  `;
}

function renderRoutineCards() {
  const today = new Date();
  const todayKey = toDateKey(today);
  const thisWeekStart = startOfWeek(today);

  const daily = activities.filter((a) => a.date === todayKey);
  const weekly = activities.filter((a) => new Date(a.date) >= thisWeekStart);

  const dailyMin = sumBy(daily, "duration");
  const dailyKm = sumBy(daily, "distance");
  const weeklyMin = sumBy(weekly, "duration");
  const weeklyKm = sumBy(weekly, "distance");

  routineCards.innerHTML = `
    <article class="routine-card"><div>Today Sessions</div><strong>${daily.length}</strong></article>
    <article class="routine-card"><div>Today Volume</div><strong>${dailyMin} min / ${dailyKm.toFixed(1)} km</strong></article>
    <article class="routine-card"><div>This Week Sessions</div><strong>${weekly.length}</strong></article>
    <article class="routine-card"><div>This Week Volume</div><strong>${weeklyMin} min / ${weeklyKm.toFixed(1)} km</strong></article>
  `;
}

function renderActivityList() {
  const top = activities.slice(0, 12);
  if (!top.length) {
    activityList.innerHTML = "<p>No activities yet.</p>";
    return;
  }

  activityList.innerHTML = top
    .map(
      (activity) => `
      <article class="log-item">
        <strong>${escapeHtml(activity.title)}</strong>
        <div class="log-meta">${activity.date} · ${capitalize(activity.category)} · ${capitalize(
        activity.entryType
      )}</div>
        <div class="log-meta">${activity.duration} min · ${activity.distance} km ${
        activity.time ? `· Time: ${escapeHtml(activity.time)}` : ""
      }</div>
      </article>
    `
    )
    .join("");
}

function renderCharts() {
  const weeklySeries = buildWeeklySeries(activities, 8);
  const labels = weeklySeries.map((x) => x.label);

  const volumeData = {
    labels,
    datasets: [
      {
        label: "Distance (km)",
        data: weeklySeries.map((x) => x.distance),
        borderColor: "#0d7b84",
        backgroundColor: "rgba(13,123,132,0.15)",
        tension: 0.3,
        fill: true
      },
      {
        label: "Duration (min)",
        data: weeklySeries.map((x) => x.duration),
        borderColor: "#f2894f",
        backgroundColor: "rgba(242,137,79,0.16)",
        tension: 0.3,
        fill: true
      }
    ]
  };

  if (volumeChart) {
    volumeChart.data = volumeData;
    volumeChart.update();
  } else {
    volumeChart = new Chart(document.getElementById("volumeChart"), {
      type: "line",
      data: volumeData,
      options: {
        responsive: true,
        plugins: { legend: { position: "bottom" } }
      }
    });
  }

  const categoryTotals = ["gym", "swimming", "running"].map((category) => {
    const entries = activities.filter((a) => a.category === category);
    return sumBy(entries, "duration");
  });

  const categoryData = {
    labels: ["Gym", "Swimming", "Running"],
    datasets: [
      {
        label: "Total minutes",
        data: categoryTotals,
        backgroundColor: ["#202124", "#0d7b84", "#f2894f"]
      }
    ]
  };

  if (categoryChart) {
    categoryChart.data = categoryData;
    categoryChart.update();
  } else {
    categoryChart = new Chart(document.getElementById("categoryChart"), {
      type: "bar",
      data: categoryData,
      options: {
        responsive: true,
        plugins: { legend: { display: false } }
      }
    });
  }
}

async function initMap(googleKey) {
  if (googleKey) {
    try {
      await loadGoogleMaps(googleKey);
      initGoogleMap();
      mapMode = "google";
      mapModeHint.textContent = "Map mode: Google Maps";
      return;
    } catch {
      mapModeHint.textContent = "Google key failed. Using OpenStreetMap fallback.";
    }
  }

  initLeafletMap();
  mapMode = "osm";
  if (!googleKey) {
    mapModeHint.textContent = "Map mode: OpenStreetMap (default)";
  }
}

function renderMapMarkers() {
  const raceActivities = activities.filter(
    (a) =>
      a.entryType === "competition" &&
      ["running", "swimming"].includes(a.category) &&
      Number.isFinite(a.lat) &&
      Number.isFinite(a.lng)
  );

  if (!raceActivities.length) {
    if (mapMode === "google") {
      googleMarkers.forEach((marker) => marker.setMap(null));
      googleMarkers = [];
    } else if (markersLayer) {
      markersLayer.clearLayers();
    }
    return;
  }

  if (mapMode === "google") {
    renderGoogleMarkers(raceActivities);
    return;
  }

  if (!markersLayer) {
    return;
  }

  markersLayer.clearLayers();
  const bounds = [];
  raceActivities.forEach((race) => {
    const marker = L.marker([race.lat, race.lng]);
    marker.bindPopup(`
      <div style="min-width:190px">
        <strong>${escapeHtml(race.title)}</strong><br/>
        <small>${escapeHtml(race.location || "Unknown location")} · ${race.date}</small>
        <hr/>
        <div>Distance: ${race.distance || "-"} km</div>
        <div>My time: ${escapeHtml(race.time || "-")}</div>
        <div>Placement: ${escapeHtml(race.placement || "-")}</div>
        ${race.officialLink ? `<a href="${escapeHtml(race.officialLink)}" target="_blank" rel="noreferrer">Official race link</a>` : ""}
      </div>
    `);

    marker.on("mouseover", () => marker.openPopup());
    marker.addTo(markersLayer);
    bounds.push([race.lat, race.lng]);
  });

  map.fitBounds(bounds, { padding: [35, 35] });
}

function initLeafletMap() {
  if (googleMap) {
    googleMap = null;
    googleMarkers = [];
  }

  if (map) {
    map.remove();
  }

  map = L.map("map", { preferCanvas: true }).setView([48.85, 2.35], 4);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(map);
  markersLayer = L.layerGroup().addTo(map);
}

function initGoogleMap() {
  if (map) {
    map.remove();
    map = null;
  }

  googleMap = new window.google.maps.Map(document.getElementById("map"), {
    center: { lat: 48.85, lng: 2.35 },
    zoom: 4,
    mapTypeControl: false,
    streetViewControl: false
  });
}

function renderGoogleMarkers(raceActivities) {
  googleMarkers.forEach((marker) => marker.setMap(null));
  googleMarkers = [];

  const infoWindow = new window.google.maps.InfoWindow();
  const bounds = new window.google.maps.LatLngBounds();

  raceActivities.forEach((race) => {
    const position = { lat: race.lat, lng: race.lng };
    const marker = new window.google.maps.Marker({
      position,
      map: googleMap,
      title: race.title
    });

    const content = `\n      <div style=\"min-width:190px\">\n        <strong>${escapeHtml(race.title)}</strong><br/>\n        <small>${escapeHtml(race.location || "Unknown location")} · ${race.date}</small>\n        <hr/>\n        <div>Distance: ${race.distance || "-"} km</div>\n        <div>My time: ${escapeHtml(race.time || "-")}</div>\n        <div>Placement: ${escapeHtml(race.placement || "-")}</div>\n        ${race.officialLink ? `<a href=\"${escapeHtml(race.officialLink)}\" target=\"_blank\" rel=\"noreferrer\">Official race link</a>` : ""}\n      </div>\n    `;

    marker.addListener("click", () => {
      infoWindow.setContent(content);
      infoWindow.open(googleMap, marker);
    });

    marker.addListener("mouseover", () => {
      infoWindow.setContent(content);
      infoWindow.open(googleMap, marker);
    });

    googleMarkers.push(marker);
    bounds.extend(position);
  });

  googleMap.fitBounds(bounds, 40);
}

function loadGoogleMaps(apiKey) {
  if (window.google && window.google.maps) {
    return Promise.resolve();
  }

  if (window._googleMapsLoadPromise) {
    return window._googleMapsLoadPromise;
  }

  window._googleMapsLoadPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}`;
    script.async = true;
    script.defer = true;
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });

  return window._googleMapsLoadPromise;
}

function renderCertificates() {
  if (!certificates.length) {
    certificateGallery.innerHTML = "<p>No certificates uploaded yet.</p>";
    return;
  }

  certificateGallery.innerHTML = certificates
    .map((cert) => {
      const media =
        cert.fileType === "pdf"
          ? `<iframe title="${escapeHtml(cert.title)}" src="${cert.dataUrl}" style="width:100%;height:120px;border:0;"></iframe>`
          : `<img src="${cert.dataUrl}" alt="${escapeHtml(cert.title)}" />`;

      return `
      <article class="certificate-card">
        ${media}
        <div class="certificate-content">
          <strong>${escapeHtml(cert.title)}</strong>
          <div class="log-meta">${escapeHtml(cert.org)}</div>
          <div class="log-meta">${cert.date}</div>
          <a href="${cert.dataUrl}" download="${escapeHtml(cert.fileName)}">Download</a>
        </div>
      </article>`;
    })
    .join("");
}

function loadFromStorage(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) {
      localStorage.setItem(key, JSON.stringify(fallback));
      return [...fallback];
    }
    return sortByDateDesc(JSON.parse(raw));
  } catch {
    return [...fallback];
  }
}

function saveToStorage(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function sortByDateDesc(list) {
  return [...list].sort((a, b) => new Date(b.date) - new Date(a.date));
}

function sumBy(arr, key) {
  return Math.round(arr.reduce((sum, item) => sum + Number(item[key] || 0), 0));
}

function parseCoordinate(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function sanitize(value) {
  return String(value).trim();
}

function capitalize(value) {
  if (!value) return "";
  return value[0].toUpperCase() + value.slice(1);
}

function toDateKey(date) {
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
}

function startOfWeek(date) {
  const d = new Date(date);
  const day = d.getDay();
  const diff = day === 0 ? 6 : day - 1;
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() - diff);
  return d;
}

function buildWeeklySeries(data, weeks) {
  const now = new Date();
  const currentWeekStart = startOfWeek(now);
  const series = [];

  for (let i = weeks - 1; i >= 0; i -= 1) {
    const weekStart = new Date(currentWeekStart);
    weekStart.setDate(weekStart.getDate() - i * 7);

    const weekEnd = new Date(weekStart);
    weekEnd.setDate(weekEnd.getDate() + 7);

    const items = data.filter((x) => {
      const d = new Date(x.date);
      return d >= weekStart && d < weekEnd;
    });

    series.push({
      label: `${String(weekStart.getMonth() + 1).padStart(2, "0")}/${String(weekStart.getDate()).padStart(
        2,
        "0"
      )}`,
      duration: items.reduce((sum, x) => sum + Number(x.duration || 0), 0),
      distance: Number(items.reduce((sum, x) => sum + Number(x.distance || 0), 0).toFixed(1))
    });
  }

  return series;
}

function calculateStreak(data) {
  const daySet = new Set(data.map((a) => a.date));
  let streak = 0;
  const cursor = new Date();

  while (daySet.has(toDateKey(cursor))) {
    streak += 1;
    cursor.setDate(cursor.getDate() - 1);
  }

  return streak;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
