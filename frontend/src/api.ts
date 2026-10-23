export type HealthResponse = {
  status: string;
  service: string;
  model_family: string;
  available_station_models: number;
};

export type StationListResponse = {
  total: number;
  limit: number;
  offset: number;
  station_ids: string[];
};

export type ForecastPoint = {
  timestamp_utc: string;
  occupancy_ratio: number;
  lower_bound: number;
  upper_bound: number;
};

export type ForecastResponse = {
  station_id: string;
  model_family: string;
  target: string;
  frequency: string;
  model_training_cutoff_utc: string;
  forecast_start_utc: string;
  forecast_end_utc: string;
  horizon_hours: number;
  points_returned: number;
  forecast: ForecastPoint[];
};

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

async function request<T>(path: string): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { Accept: "application/json" },
    });
  } catch {
    throw new Error(
      `Cannot reach the CycleCast API at ${API_BASE_URL}. Start the FastAPI backend and check its address.`,
    );
  }

  if (!response.ok) {
    let message = `API request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // Keep the HTTP status message when the response is not JSON.
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export const api = {
  getHealth: () => request<HealthResponse>("/health"),
  getStations: (limit = 1000, offset = 0) =>
    request<StationListResponse>(`/stations?limit=${limit}&offset=${offset}`),
  getForecast: (stationId: string, horizonHours: number) =>
    request<ForecastResponse>(
      `/forecast/${encodeURIComponent(stationId)}?horizon_hours=${horizonHours}`,
    ),
};

export const apiBaseUrl = API_BASE_URL;
