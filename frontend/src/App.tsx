import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  Bike,
  CalendarClock,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  ExternalLink,
  Gauge,
  Layers3,
  LoaderCircle,
  RefreshCw,
  Server,
  ShieldCheck,
  SlidersHorizontal,
  Waves,
} from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  api,
  apiBaseUrl,
  type ForecastPoint,
  type ForecastResponse,
  type HealthResponse,
} from "./api";

type LoadState = "idle" | "loading" | "success" | "error";

function formatDate(value?: string) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short",
  }).format(date);
}

function formatClock(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  }).format(date);
}

function percentage(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ payload?: ForecastPoint & { timeLabel?: string } }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0]?.payload;
  if (!point) return null;

  return (
    <div className="chart-tooltip">
      <p className="tooltip-time">{point.timeLabel || label}</p>
      <div className="tooltip-row">
        <span className="tooltip-dot" />
        <span>Predicted occupancy</span>
        <strong>{percentage(point.occupancy_ratio)}</strong>
      </div>
      <div className="tooltip-row muted">
        <span>Expected range</span>
        <strong>
          {percentage(point.lower_bound)} – {percentage(point.upper_bound)}
        </strong>
      </div>
    </div>
  );
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stationIds, setStationIds] = useState<string[]>([]);
  const [stationId, setStationId] = useState("");
  const [horizon, setHorizon] = useState(24);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [healthState, setHealthState] = useState<LoadState>("loading");
  const [forecastState, setForecastState] = useState<LoadState>("idle");
  const [error, setError] = useState("");
  const [stationSearch, setStationSearch] = useState("");

  const loadInitialData = useCallback(async () => {
    setHealthState("loading");
    try {
      const [healthResult, stationsResult] = await Promise.all([
        api.getHealth(),
        api.getStations(1000, 0),
      ]);
      setHealth(healthResult);
      setStationIds(stationsResult.station_ids);
      setStationId((current) =>
        current && stationsResult.station_ids.includes(current)
          ? current
          : stationsResult.station_ids[0] || "",
      );
      setHealthState("success");
    } catch (cause) {
      setHealthState("error");
      setError(cause instanceof Error ? cause.message : "Unable to load API status.");
    }
  }, []);

  const loadForecast = useCallback(async () => {
    if (!stationId) return;
    setForecastState("loading");
    setError("");
    try {
      const result = await api.getForecast(stationId, horizon);
      setForecast(result);
      setForecastState("success");
    } catch (cause) {
      setForecastState("error");
      setError(cause instanceof Error ? cause.message : "Forecast request failed.");
    }
  }, [stationId, horizon]);

  useEffect(() => {
    void loadInitialData();
  }, [loadInitialData]);

  useEffect(() => {
    if (stationId && healthState === "success") {
      void loadForecast();
    }
  }, [stationId, horizon, healthState, loadForecast]);

  const chartData = useMemo(
    () =>
      (forecast?.forecast ?? []).map((point) => ({
        ...point,
        timeLabel: formatClock(point.timestamp_utc),
      })),
    [forecast],
  );

  const summary = useMemo(() => {
    if (!forecast?.forecast.length) return null;
    const values = forecast.forecast.map((point) => point.occupancy_ratio);
    const average = values.reduce((sum, value) => sum + value, 0) / values.length;
    const peak = Math.max(...values);
    const lowest = Math.min(...values);
    return { average, peak, lowest };
  }, [forecast]);

  const filteredStations = useMemo(
    () =>
      stationIds.filter((id) =>
        id.toLowerCase().includes(stationSearch.toLowerCase().trim()),
      ),
    [stationIds, stationSearch],
  );

  const apiOnline = healthState === "success";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="CycleCast home">
          <span className="brand-mark"><Waves size={23} strokeWidth={2.4} /></span>
          <span className="brand-copy">
            <strong>cyclecast</strong>
            <small>FORECASTING PLATFORM</small>
          </span>
        </a>

        <div className="workspace-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          <a className="nav-item active" href="#overview">
            <Gauge size={18} />
            <span>Forecast overview</span>
            <span className="nav-indicator" />
          </a>
          <a className="nav-item" href="#forecast">
            <Activity size={18} />
            <span>Station forecasts</span>
          </a>
          <a className="nav-item" href="#model-info">
            <Layers3 size={18} />
            <span>Model information</span>
          </a>
        </nav>

        <div className="sidebar-spacer" />

        <div className="sidebar-status">
          <div className="sidebar-status-top">
            <span className={`status-dot ${apiOnline ? "online" : ""}`} />
            <span>{apiOnline ? "API connected" : healthState === "loading" ? "Connecting…" : "API unavailable"}</span>
          </div>
          <p>Local development environment</p>
          <button className="text-button" onClick={() => void loadInitialData()}>
            <RefreshCw size={13} /> Refresh connection
          </button>
        </div>

        <div className="sidebar-footer">
          <span className="footer-logo"><Bike size={15} /></span>
          <span>CycleCast ML Platform</span>
          <span className="version-pill">v1.0</span>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>Workspace</span><span className="crumb-slash">/</span><strong>Forecast overview</strong>
          </div>
          <div className="topbar-right">
            <span className="environment-pill"><span className="status-dot online" /> LOCAL ENVIRONMENT</span>
            <div className="avatar">CC</div>
          </div>
        </header>

        <div className="page-content">
          <section className="page-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> BICYCLE MOBILITY INTELLIGENCE</div>
              <h1>Forecast overview</h1>
              <p className="page-subtitle">
                Explore station occupancy forecasts powered by your trained Prophet models.
              </p>
            </div>
            <button className="button button-secondary" onClick={() => { void loadInitialData(); if (stationId) void loadForecast(); }}>
              <RefreshCw size={15} /> Refresh data
            </button>
          </section>

          <section className="notice-banner" aria-label="Historical data notice">
            <div className="notice-icon"><CircleHelp size={18} /></div>
            <div className="notice-copy">
              <strong>Historical model · Not live availability</strong>
              <p>Models were trained on May 2024 observations. Forecast timestamps begin after each model’s training cutoff and should not be interpreted as current station conditions.</p>
            </div>
            <span className="notice-tag">HISTORICAL</span>
          </section>

          <section className="metrics-grid" aria-label="Platform metrics">
            <MetricCard
              icon={<Server size={17} />}
              label="API status"
              value={apiOnline ? "Connected" : healthState === "loading" ? "Checking" : "Offline"}
              note={apiOnline ? "FastAPI responding" : "Check backend service"}
              tone={apiOnline ? "green" : "amber"}
              stateIcon={apiOnline ? <Check size={13} /> : <AlertCircle size={13} />}
            />
            <MetricCard
              icon={<Bike size={17} />}
              label="Available station models"
              value={health ? health.available_station_models.toLocaleString("en-US") : "—"}
              note="Saved Prophet artifacts"
              tone="blue"
              stateIcon={<Layers3 size={13} />}
            />
            <MetricCard
              icon={<Activity size={17} />}
              label="Forecast resolution"
              value="15 min"
              note="96 points per 24 hours"
              tone="violet"
              stateIcon={<Clock3 size={13} />}
            />
            <MetricCard
              icon={<ShieldCheck size={17} />}
              label="Forecast model"
              value="Prophet"
              note="Occupancy ratio · 0–100%"
              tone="teal"
              stateIcon={<Check size={13} />}
            />
          </section>

          <section className="control-panel" aria-label="Forecast controls">
            <div className="panel-heading">
              <div className="panel-title-icon"><SlidersHorizontal size={17} /></div>
              <div>
                <h2>Forecast configuration</h2>
                <p>Choose a station and prediction horizon.</p>
              </div>
              <span className="panel-step">CONFIGURE</span>
            </div>
            <div className="control-grid">
              <div className="field-group station-field">
                <label htmlFor="station-search">Station</label>
                <div className="station-select-wrap">
                  <Bike size={17} className="field-leading-icon" />
                  <input
                    id="station-search"
                    type="text"
                    value={stationSearch}
                    onChange={(event) => setStationSearch(event.target.value)}
                    placeholder={stationId || "Search station IDs…"}
                    aria-label="Search station IDs"
                  />
                  <ChevronDown size={16} className="field-trailing-icon" />
                </div>
                <div className="station-options" role="group" aria-label="Available stations">
                  {filteredStations.slice(0, 6).map((id) => (
                    <button
                      key={id}
                      className={`station-option ${stationId === id ? "selected" : ""}`}
                      onClick={() => { setStationId(id); setStationSearch(""); }}
                      type="button"
                    >
                      <span className="station-option-dot" />
                      <span>{id}</span>
                      {stationId === id && <Check size={14} />}
                    </button>
                  ))}
                  {filteredStations.length === 0 && (
                    <div className="station-empty">{stationIds.length ? "No matching station IDs" : "No stations loaded"}</div>
                  )}
                </div>
                <span className="field-help">{stationIds.length.toLocaleString("en-US")} station models discovered</span>
              </div>

              <div className="field-group horizon-field">
                <label htmlFor="horizon">Forecast horizon</label>
                <div className="select-wrap">
                  <CalendarClock size={17} className="field-leading-icon" />
                  <select id="horizon" value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}>
                    <option value={6}>Next 6 hours</option>
                    <option value={12}>Next 12 hours</option>
                    <option value={24}>Next 24 hours</option>
                    <option value={48}>Next 48 hours</option>
                    <option value={72}>Next 72 hours</option>
                    <option value={168}>Next 7 days</option>
                  </select>
                  <ChevronDown size={16} className="field-trailing-icon" />
                </div>
                <span className="field-help">{horizon * 4} forecast intervals at 15-minute resolution</span>
              </div>

              <div className="control-action">
                <button
                  className="button button-primary"
                  onClick={() => void loadForecast()}
                  disabled={!stationId || forecastState === "loading"}
                >
                  {forecastState === "loading" ? <LoaderCircle className="spin" size={16} /> : <Activity size={16} />}
                  Generate forecast
                </button>
                <span className="action-note"><ShieldCheck size={13} /> Uses saved model · No retraining</span>
              </div>
            </div>
          </section>

          {error && (
            <div className="error-banner" role="alert">
              <AlertCircle size={18} />
              <div><strong>Unable to complete request</strong><p>{error}</p></div>
              <button className="icon-button" onClick={() => setError("")} aria-label="Dismiss error">×</button>
            </div>
          )}

          <section className="chart-panel" id="forecast">
            <div className="chart-panel-header">
              <div>
                <div className="section-kicker">PREDICTION ANALYTICS</div>
                <h2>Occupancy forecast</h2>
                <p className="panel-description">
                  {forecast ? `${forecast.station_id} · ${forecast.horizon_hours}-hour outlook` : "Select a station to load its forecast."}
                </p>
              </div>
              <div className="chart-legend">
                <span><i className="legend-line" /> Predicted occupancy</span>
                <span><i className="legend-area" /> Uncertainty range</span>
              </div>
            </div>

            <div className="chart-meta">
              <span><span className="status-dot online" /> {forecastState === "success" ? "Forecast ready" : forecastState === "loading" ? "Generating forecast" : forecastState === "error" ? "Forecast unavailable" : "Awaiting request"}</span>
              {forecast && <span><Clock3 size={13} /> {forecast.points_returned} data points</span>}
              {forecast && <span><CalendarClock size={13} /> Cutoff: {formatDate(forecast.model_training_cutoff_utc)}</span>}
            </div>

            <div className="chart-container">
              {forecastState === "loading" ? (
                <div className="chart-empty"><LoaderCircle className="spin" size={25} /><strong>Generating forecast…</strong><span>Requesting predictions from the Prophet API.</span></div>
              ) : chartData.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} margin={{ top: 10, right: 14, left: 0, bottom: 8 }}>
                    <defs>
                      <linearGradient id="confidenceFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#56a7f8" stopOpacity={0.23} />
                        <stop offset="100%" stopColor="#56a7f8" stopOpacity={0.025} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#e9edf3" strokeDasharray="3 5" vertical={false} />
                    <XAxis
                      dataKey="timeLabel"
                      tick={{ fill: "#8993a5", fontSize: 11 }}
                      tickLine={false}
                      axisLine={{ stroke: "#e6eaf0" }}
                      minTickGap={42}
                    />
                    <YAxis
                      domain={[0, 1]}
                      tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
                      tick={{ fill: "#8993a5", fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      width={48}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="upper_bound"
                      stroke="none"
                      fill="url(#confidenceFill)"
                      activeDot={false}
                      isAnimationActive={false}
                    />
                    <Area
                      type="monotone"
                      dataKey="lower_bound"
                      stroke="none"
                      fill="#ffffff"
                      activeDot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="occupancy_ratio"
                      stroke="#2584e8"
                      strokeWidth={2.4}
                      dot={false}
                      activeDot={{ r: 4, fill: "#2584e8", stroke: "#fff", strokeWidth: 2 }}
                      isAnimationActive={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <div className="chart-empty">
                  <div className="chart-empty-icon"><Activity size={25} /></div>
                  <strong>{forecastState === "error" ? "Forecast could not be loaded" : "Your forecast will appear here"}</strong>
                  <span>{healthState === "error" ? "Start the FastAPI backend and refresh the connection." : "Choose a station and forecast horizon, then generate a forecast."}</span>
                  {healthState === "error" && <code>{apiBaseUrl}</code>}
                </div>
              )}
            </div>
            <div className="chart-footer">
              <span><span className="footer-dot" /> Occupancy ratio (0–1)</span>
              <span>Forecast timestamps are UTC</span>
            </div>
          </section>

          <section className="lower-grid">
            <div className="summary-panel">
              <div className="lower-panel-heading">
                <div><h2>Forecast summary</h2><p>Computed from the returned predictions</p></div>
                <span className="small-icon-box"><Gauge size={17} /></span>
              </div>
              {summary ? (
                <div className="summary-metrics">
                  <div className="summary-metric">
                    <span>Mean occupancy</span>
                    <strong>{percentage(summary.average)}</strong>
                    <div className="mini-track"><span style={{ width: `${summary.average * 100}%` }} /></div>
                  </div>
                  <div className="summary-metric">
                    <span>Peak predicted</span>
                    <strong>{percentage(summary.peak)}</strong>
                    <div className="mini-track peak"><span style={{ width: `${summary.peak * 100}%` }} /></div>
                  </div>
                  <div className="summary-metric">
                    <span>Lowest predicted</span>
                    <strong>{percentage(summary.lowest)}</strong>
                    <div className="mini-track low"><span style={{ width: `${summary.lowest * 100}%` }} /></div>
                  </div>
                </div>
              ) : (
                <div className="empty-summary"><Activity size={20} /><span>Summary metrics appear after a forecast is generated.</span></div>
              )}
              <div className="summary-footnote"><CircleHelp size={14} /> Values are calculated from clipped occupancy predictions.</div>
            </div>

            <div className="model-panel" id="model-info">
              <div className="lower-panel-heading">
                <div><h2>Model information</h2><p>Active inference configuration</p></div>
                <span className="small-icon-box"><Layers3 size={17} /></span>
              </div>
              <div className="model-info-list">
                <InfoRow label="Model family" value="Prophet" />
                <InfoRow label="Forecast target" value="Occupancy ratio" />
                <InfoRow label="Sampling interval" value="15 minutes" />
                <InfoRow label="Output bounds" value="0.0 – 1.0" />
                <InfoRow label="Training cutoff" value={forecast ? formatDate(forecast.model_training_cutoff_utc) : "—"} />
              </div>
              <a className="api-docs-link" href={`${apiBaseUrl}/docs`} target="_blank" rel="noreferrer">
                Open API documentation <ExternalLink size={14} />
              </a>
            </div>
          </section>

          <section className="table-panel">
            <div className="table-heading">
              <div><h2>Forecast data</h2><p>First 10 intervals from the selected prediction horizon</p></div>
              <span className="table-count">{forecast ? `${Math.min(10, forecast.forecast.length)} of ${forecast.forecast.length} points` : "No data"}</span>
            </div>
            <div className="table-scroll">
              <table>
                <thead><tr><th>Timestamp (UTC)</th><th>Predicted occupancy</th><th>Lower bound</th><th>Upper bound</th><th>Signal</th></tr></thead>
                <tbody>
                  {forecast?.forecast.slice(0, 10).map((point) => (
                    <tr key={point.timestamp_utc}>
                      <td className="timestamp-cell">{formatDate(point.timestamp_utc)}</td>
                      <td><strong>{percentage(point.occupancy_ratio)}</strong></td>
                      <td className="muted-cell">{percentage(point.lower_bound)}</td>
                      <td className="muted-cell">{percentage(point.upper_bound)}</td>
                      <td><span className={`signal-pill ${point.occupancy_ratio >= 0.75 ? "high" : point.occupancy_ratio <= 0.25 ? "low" : "moderate"}`}><span />{point.occupancy_ratio >= 0.75 ? "High" : point.occupancy_ratio <= 0.25 ? "Low" : "Moderate"}</span></td>
                    </tr>
                  ))}
                  {!forecast && <tr><td colSpan={5} className="empty-table">Generate a forecast to inspect prediction values.</td></tr>}
                </tbody>
              </table>
            </div>
          </section>

          <footer className="page-footer">
            <span>© {new Date().getFullYear()} CycleCast · Bicycle occupancy forecasting</span>
            <span><ShieldCheck size={13} /> Local API · Historical model artifacts</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  note,
  tone,
  stateIcon,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  note: string;
  tone: string;
  stateIcon: React.ReactNode;
}) {
  return (
    <article className="metric-card">
      <div className="metric-top">
        <span className={`metric-icon ${tone}`}>{icon}</span>
        <span className={`metric-trend ${tone}`}>{stateIcon}</span>
      </div>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      <span className="metric-note">{note}</span>
    </article>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return <div className="info-row"><span>{label}</span><strong>{value}</strong></div>;
}

export default App;
