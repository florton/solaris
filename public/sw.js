// solaris offline strategy:
// - navigations: network-first (never serve stale HTML against gone hashed
//   assets), cache the fresh copy, fall back to cache when offline.
// - everything else (hashed js/css, model weights, wasm): cache-first,
//   then network + cache-fill. Weights are frozen with the build; bump CACHE
//   when they change and old caches are purged on activate.
const CACHE = 'solaris-v3'; // bump: added models/imagination/

self.addEventListener('install', (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('message', (event) => {
  if (event.data?.type !== 'precache') return;
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      const results = await Promise.allSettled(event.data.urls.map((u) => cache.add(u)));
      const ok = results.filter((r) => r.status === 'fulfilled').length;
      if (event.source) event.source.postMessage({ type: 'precache-done', ok, total: results.length });
    })(),
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== location.origin) return;

  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .then((res) => {
          if (res.ok) {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put('./', copy));
          }
          return res;
        })
        .catch(() =>
          caches.match('./', { ignoreSearch: true }).then((hit) => hit ?? Response.error()),
        ),
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then(
      (hit) =>
        hit ??
        fetch(event.request).then((res) => {
          if (res.ok) {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put(event.request, copy));
          }
          return res;
        }),
    ),
  );
});
