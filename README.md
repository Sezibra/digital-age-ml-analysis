# TriLog - Personal Sports Website

A single-page website to track your gym, swimming, and running sessions, analyze daily/weekly routines, and display race history on a map.

## Features

- Add training and competition entries (date, duration, distance, time, placement, notes).
- Daily/weekly routine cards.
- Trend charts (8-week volume + category totals).
- Competition map with marker popups showing:
  - Official race link
  - Your time
  - Distance
  - Placement
- Certificate upload gallery (images and PDFs).
- Data persistence with browser `localStorage`.

## Files

- `index.html` - page structure
- `styles.css` - UI styling and responsive layout
- `script.js` - app logic, charts, map, and storage

## Run

Open `/Users/cansezgin/Documents/New project/index.html` in a browser.

## Google Maps

The app works immediately with OpenStreetMap.

If you want Google Maps:

1. Create/get a Google Maps JavaScript API key in Google Cloud.
2. Paste it in the `Google Maps API Key` field on the page.
3. Click `Use Google Map`.

The key is stored locally in your browser only.

## Creative additions you can add next

- Goal planning (weekly distance target + progress bar)
- PB tracker (auto-highlight personal bests)
- Medal board (podium finishes and badges)
- Export to CSV / PDF race report
- Public read-only mode + private edit mode with password
