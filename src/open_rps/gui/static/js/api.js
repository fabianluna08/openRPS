// Comunicación con el servidor local (open_rps/gui/server.py).

async function request(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    throw new Error("No hay conexión con el servidor de OpenRPS. ¿Sigue abierta la consola de open-rps-gui?");
  }
  let data;
  try {
    data = await response.json();
  } catch (error) {
    throw new Error(`Respuesta inválida del servidor (${response.status})`);
  }
  if (!response.ok) throw new Error(data && data.error ? data.error : `Error ${response.status}`);
  return data;
}

export const api = {
  info: () => request("/api/info"),
  initialProject: () => request("/api/project"),
  example: (name) => request(`/api/example/${encodeURIComponent(name)}`),
  validate: (project) => request("/api/validate", { project }),
  simulate: (project) => request("/api/simulate", { project }),
  probe: (project, point) => request("/api/probe", { project, point }),
  parseMeasurements: (project, csv) => request("/api/measurements/parse", { project, csv }),
  compare: (project) => request("/api/compare", { project }),
  calibrate: (project, options) => request("/api/calibrate", { project, options }),
  planChannels: (project, options, resultId) => request("/api/wifi/plan", { project, options, result_id: resultId }),
  exportUrl: (resultId, what) => `/api/export/${encodeURIComponent(resultId)}/${encodeURIComponent(what)}`,
};

// Rejilla float32 little-endian en base64 -> Float32Array (NaN e infinitos se conservan)
export function decodeGrid(grid) {
  const binary = atob(grid.data);
  const bytes = new Uint8Array(binary.length);
  for (let k = 0; k < binary.length; k++) bytes[k] = binary.charCodeAt(k);
  const values = new Float32Array(bytes.buffer);
  return { ny: grid.shape[0], nx: grid.shape[1], values };
}

export function decodeResult(payload) {
  const metrics = {};
  for (const [name, grid] of Object.entries(payload.metrics)) metrics[name] = decodeGrid(grid).values;
  const perTx = payload.per_tx_rx_power ? payload.per_tx_rx_power.map((g) => decodeGrid(g).values) : null;
  return { ...payload, metrics, perTx };
}
