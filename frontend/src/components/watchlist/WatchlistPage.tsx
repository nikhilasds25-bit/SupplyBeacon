import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Bell, Boxes, Factory, Search, Warehouse as WarehouseIcon, X } from 'lucide-react'
import { getCurrentImpact, getCurrentRiskWatchlist, getCurrentSnapshot } from '../../services/supplyBeaconApi'
import type { Entity, Event, ImpactTrace, RiskWatchlistRow, Snapshot } from '../../services/supplyBeaconApi'
import RiskRanking from './RiskRanking'

type WatchItem = { id: string; type: 'Product' | 'Warehouse' | 'Supplier' }
type Filter = 'All' | 'Affected' | 'Not affected' | 'Product' | 'Warehouse' | 'Supplier' | 'Medium' | 'High' | 'Critical'

const STORAGE_KEY = 'supplybeacon-watchlist'
const text = (value: unknown, fallback = 'Unavailable') => value === undefined || value === null || value === '' ? fallback : String(value)
const idOf = (entity: Entity) => text(entity.node_id || entity.Node_ID)
const riskOf = (entity?: Entity) => text(entity?.risk_level, 'Unavailable')
const typeIcon = (type: WatchItem['type']) => type === 'Product' ? <Boxes /> : type === 'Warehouse' ? <WarehouseIcon /> : <Factory />

function entityAffectedByTrace(entity: Entity, trace: ImpactTrace) {
  const id = idOf(entity)
  return entity.node_type === 'Product' ? trace.product_ids.includes(id) : entity.node_type === 'Warehouse' ? trace.warehouses.some(warehouse => warehouse.node_id === id) : trace.supplier_ids.includes(id)
}
function entityAffected(entity: Entity, impact: ImpactTrace[]) {
  return impact.some(trace => entityAffectedByTrace(entity, trace))
}
function priorityForTrace(trace?: ImpactTrace) {
  if (!trace) return 'INFORMATION'
  const levels = trace.network_risk_levels || trace.risk_levels || {}
  if ((levels.Critical || 0) > 0) return 'CRITICAL'
  if ((levels.High || 0) > 0) return 'HIGH'
  if ((levels.Medium || 0) > 0) return 'MEDIUM'
  return 'INFORMATION'
}
const temporalPriority: Record<string, number> = { ACTIVE: 0, UPCOMING: 1, EXPIRED: 2, UNKNOWN: 3 }
const severityPriority: Record<string, number> = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, INFORMATION: 3 }

function Details({ entity, traces, onClose }: { entity: Entity; traces: ImpactTrace[]; onClose: () => void }) {
  const matching = traces.filter(trace => entityAffectedByTrace(entity, trace))
  const warehouse = text(entity.warehouse)
  const supplier = text(entity.supplier || entity.brand)
  const pathLabel = entity.node_type === 'Supplier' ? 'Supplier → SUPPLIES → Brand → PRODUCES → Product → STORED_AT → Warehouse → optional LOCATED_IN → ACTIVE Event' : entity.node_type === 'Warehouse' ? 'Warehouse → optional LOCATED_IN → ACTIVE Event' : 'Product → STORED_AT → Warehouse → optional LOCATED_IN → ACTIVE Event'
  return <div className="watch-modal-backdrop" onClick={onClose}><div className="watch-modal" role="dialog" aria-modal="true" onClick={event => event.stopPropagation()}><div className="analytics-modal-head"><div><p className="eyebrow">WATCHED {text(entity.node_type).toUpperCase()}</p><h2>{text(entity.name || entity.Node_Name)}</h2></div><button className="icon-btn" onClick={onClose} aria-label="Close"><X /></button></div><div className="watch-detail-grid"><span>Risk level<strong>{riskOf(entity)}</strong></span><span>Scenario score<strong>{entity.risk_score === undefined ? 'Unavailable' : String(entity.risk_score)}</strong></span><span>Warehouse<strong>{warehouse}</strong></span><span>Supplier / brand<strong>{supplier}</strong></span><span>Risk reason<strong>{text(entity.risk_reason || entity.Risk_Reason)}</strong></span></div><h3>WHY IS THIS AFFECTED?</h3>{matching.length ? matching.map(trace => <div className="watch-path" key={String(trace.event.event_id)}><p><b>  {pathLabel}</b></p><p>{text(trace.event.title || trace.event.event_type)} · {text(trace.event.location)}</p><small><b>OBSERVED</b> external source · <b>DERIVED</b> event eligibility and scenario risk · <b>MODELED</b> network relationships</small></div>) : <p className="muted">No active operational event currently affects this watched entity.</p>}</div></div>
}

function AlertCard({ event, trace, onInspect }: { event: Event; trace?: ImpactTrace; onInspect: () => void }) {
  const status = text(event.event_status, 'UNKNOWN')
  const priority = priorityForTrace(trace)
  return <div className="watch-alert"><div className="watch-alert-status"><Bell />{status}</div><div><strong>{text(event.title || event.event_type)}</strong><p>{trace ? `${trace.product_ids.length} products exposed · ${trace.supplier_ids.length} related suppliers` : text(event.location, 'Location unavailable')}</p><small>Source: {text(event.source_channel)} · Status: {status} · Valid until: {text(event.event_end)}</small></div><span className="watch-priority">{priority}</span>{trace && <button className="secondary compact-btn" onClick={onInspect}>Inspect Impact</button>}</div>
}

function ImpactDetail({ trace, onClose }: { trace: ImpactTrace; onClose: () => void }) {
  const event = trace.event
  return <div className="watch-modal-backdrop" onClick={onClose}><div className="watch-modal" role="dialog" aria-modal="true" onClick={click => click.stopPropagation()}><div className="analytics-modal-head"><div><p className="eyebrow">EVIDENCE TO IMPACT</p><h2>{text(event.title || event.event_type)}</h2></div><button className="icon-btn" onClick={onClose} aria-label="Close"><X /></button></div><div className="watch-detail-grid"><span>Event type<strong>{text(event.event_type)}</strong></span><span>Source<strong>{text(event.source_channel)}</strong></span><span>Status<strong>{text(event.event_status)}</strong></span><span>Location<strong>{text(event.location)}</strong></span><span>Valid until<strong>{text(event.event_end)}</strong></span><span>Graph event<strong>{trace.graph_event_node_id}</strong></span></div><div className="watch-path"><p><b>OBSERVED SOURCE</b> → <b>EVENT</b> → <b>LOCATION</b> → <b>WAREHOUSE</b> → <b>PRODUCTS</b> → <b>SUPPLIERS</b></p><p>{text(event.source_channel)} → {text(event.event_type)} → {text(event.location)} → {trace.warehouses.map(warehouse => warehouse.name).join(', ') || 'Unavailable'} → {trace.product_ids.length} products → {trace.supplier_ids.length} suppliers</p></div><div className="watch-detail-grid"><span>Medium exposure<strong>{trace.risk_levels.Medium || 0}</strong></span><span>High exposure<strong>{trace.risk_levels.High || 0}</strong></span><span>Critical exposure<strong>{trace.risk_levels.Critical || 0}</strong></span></div><p className="muted">{trace.provenance}</p></div></div>
}

function PriorityWatchlist() {
  const [rows, setRows] = useState<RiskWatchlistRow[] | null>(null)
  useEffect(() => { getCurrentRiskWatchlist().then(setRows).catch(() => setRows([])) }, [])
  return <section className="card"><div className="card-title"><div><p className="eyebrow">RISK WATCHLIST</p><h3>Warehouse priority now</h3></div><span className="badge">DERIVED / RULE-BASED</span></div>{rows === null ? <p className="muted">Loading evidence-ranked warehouses…</p> : rows.length === 0 ? <p className="muted">Watchlist evidence unavailable.</p> : <div className="watch-table">{rows.slice(0, 15).map(row => <div className="watch-row" key={row.entity_id}><span className="watch-entity-icon"><WarehouseIcon /></span><span><strong>{row.entity_name}</strong><small>{row.location || 'Location unavailable'} · {row.reason}</small></span><span>{row.risk_level}</span><span>{row.risk_score === null ? 'Unavailable' : row.risk_score.toFixed(3)}</span><span>{row.operational_event_count} operational · {row.active_signal_count} active signals</span><span>{row.risk_availability}</span><span>{row.top_event || 'No active operational event'}</span></div>)}</div>}</section>
}

function SavedWatchlistPage() {
  const [data, setData] = useState<Snapshot | null>(null)
  const [impact, setImpact] = useState<ImpactTrace[]>([])
  const [items, setItems] = useState<WatchItem[]>(() => { try { const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); return Array.isArray(value) ? value : [] } catch { return [] } })
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<Filter>('All')
  const [selected, setSelected] = useState<Entity | null>(null); const [selectedImpact, setSelectedImpact] = useState<ImpactTrace | null>(null)
  const [error, setError] = useState(false); const [impactLoaded, setImpactLoaded] = useState(false)
  useEffect(() => { getCurrentSnapshot().then(setData).catch(() => setError(true)); getCurrentImpact().then(value => { setImpact(value); setImpactLoaded(true) }).catch(() => setImpactLoaded(false)) }, [])
  const persistWatchlist = (nextItems: WatchItem[]) => { setItems(nextItems); localStorage.setItem(STORAGE_KEY, JSON.stringify(nextItems)); window.dispatchEvent(new Event('supplybeacon-watchlist-changed')) }
  const entities = data?.entities || []
  const watched = useMemo(() => items.map(item => entities.find(entity => idOf(entity) === item.id && entity.node_type === item.type)).filter(Boolean) as Entity[], [items, entities])
  const candidates = entities.filter(entity => ['Product', 'Warehouse', 'Supplier'].includes(String(entity.node_type)) && text(entity.name).toLowerCase().includes(query.toLowerCase()) && !items.some(item => item.id === idOf(entity)))
  const affected = (entity: Entity): boolean | null => impactLoaded ? entityAffected(entity, impact) : null
  const visible = watched.filter(entity => filter === 'All' || filter === 'Affected' && affected(entity) === true || filter === 'Not affected' && affected(entity) === false || filter === entity.node_type || filter === riskOf(entity))
  const activeTraces = impact.filter(trace => trace.event.event_status === 'ACTIVE')
  const activeWatchAlerts = impactLoaded ? activeTraces.filter(trace => watched.some(entity => entityAffectedByTrace(entity, trace))) : []
  const add = (entity: Entity) => { persistWatchlist([...items, { id: idOf(entity), type: entity.node_type as WatchItem['type'] }]); setQuery('') }
  const remove = (entity: Entity) => persistWatchlist(items.filter(item => item.id !== idOf(entity)))
  if (!data && !error) return <div className="loading">Loading risk watchlist…</div>
  if (error || !data) return <section className="card empty-card"><AlertTriangle /><h3>CURRENT DATA UNAVAILABLE</h3><p className="muted">Core current-state data is unavailable. Watchlist entries remain stored locally.</p></section>
  return <div className="watchlist-page"><div className="page-heading"><div><p className="eyebrow">MONITOR</p><h1>Risk Watchlist</h1><p className="muted">Monitor selected entities against validated current operational exposure.</p></div><span className="badge">{activeWatchAlerts.length} ACTIVE WATCH ALERTS</span></div><div className="watch-summary-grid"><div className="card"><small>WATCHED ENTITIES</small><strong>{watched.length}</strong></div><div className="card"><small>CURRENTLY AFFECTED</small><strong>{  impactLoaded ? watched.filter(entity => affected(entity) === true).length : 'Unavailable'}</strong></div><div className="card"><small>MEDIUM RISK</small><strong>{watched.filter(entity => riskOf(entity) === 'Medium').length}</strong></div><div className="card"><small>HIGH / CRITICAL</small><strong>{watched.filter(entity => ['High', 'Critical'].includes(riskOf(entity))).length}</strong></div><div className="card"><small>ACTIVE EVENTS AFFECTING WATCHLIST</small><strong>{activeWatchAlerts.length}</strong></div></div><section className="card watch-add"><div className="card-title"><div><p className="eyebrow">ADD ENTITY</p><h3>Choose a monitored entity</h3></div></div><div className="watch-search"><Search /><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Search products, warehouses or suppliers…" /></div>{query && <div className="watch-candidates">{candidates.slice(0, 8).map(entity => <button key={idOf(entity)} onClick={() => add(entity)}>{typeIcon(entity.node_type as WatchItem['type'])}<span>{text(entity.name)}<small>{entity.node_type} · {riskOf(entity)}</small></span></button>)}</div>}</section><section className="card"><div className="card-title"><div><p className="eyebrow">WATCHED ENTITIES</p><h3>Current exposure</h3></div><div className="watch-filters">{(['All', 'Affected', 'Not affected', 'Product', 'Warehouse', 'Supplier', 'Medium', 'High', 'Critical'] as Filter[]).map(value => <button className={filter === value ? 'active' : ''} key={value} onClick={() => setFilter(value)}>{value}</button>)}</div></div>{visible.length ? <div className="watch-table">{visible.map(entity => <button className="watch-row" key={idOf(entity)} onClick={() => setSelected(entity)}><span className="watch-entity-icon">{typeIcon(entity.node_type as WatchItem['type'])}</span><span><strong>{text(entity.name)}</strong><small>{entity.node_type} · {affected(entity) === null ? 'Unavailable' : affected(entity) ? 'Affected' : 'Not affected'}</small></span><span>{riskOf(entity)}</span><span>{entity.risk_score == null ? 'Unavailable' : Number(entity.risk_score).toFixed(2)}</span><span>{  impactLoaded ? impact.filter(trace => entityAffectedByTrace(entity, trace) && trace.event.event_status === 'ACTIVE').length : 'Unavailable'} active events</span><span>DERIVED / RULE_BASED</span><span><button className="watch-remove" onClick={event => { event.stopPropagation(); remove(entity) }}>Remove</button></span></button>)}</div> : <p className="muted">No watched entities match this filter.</p>}</section><section className="card"><div className="card-title"><div><p className="eyebrow">ALERT CENTER</p><h3>Current evidence alerts</h3></div><span className="badge">{activeWatchAlerts.length} relevant</span></div>{  data.events.filter(event => ['ACTIVE', 'UPCOMING', 'EXPIRED', 'UNKNOWN'].includes(String(event.event_status))).sort((a, b) => { const traceA = impact.find(item => String(item.event.event_id) === String(a.event_id)); const traceB = impact.find(item => String(item.event.event_id) === String(b.event_id)); return (severityPriority[priorityForTrace(traceA)] - severityPriority[priorityForTrace(traceB)]) || ((temporalPriority[String(a.event_status)] ?? 3) - (temporalPriority[String(b.event_status)] ?? 3)) }).map(event => { const trace = activeTraces.find(item => String(item.event.event_id) === String(event.event_id)); return <AlertCard key={String(event.event_id)} event={event} trace={trace} onInspect={() => trace && setSelectedImpact(trace)} /> })}</section>{selected && <Details entity={selected} traces={impact} onClose={() => setSelected(null)} />}{selectedImpact && <ImpactDetail trace={selectedImpact} onClose={() => setSelectedImpact(null)} />}</div>
}

export default function WatchlistPage() {
  return <><RiskRanking /><SavedWatchlistPage /></>
}
