import {useEffect, useMemo, useState} from 'react'
import {Link} from 'react-router-dom'
import {getCurrentRiskWatchlist} from '../../services/supplyBeaconApi'
import type {RiskWatchlistRow} from '../../services/supplyBeaconApi'

type Kind = 'Warehouse' | 'Product' | 'Supplier'
const display = (value: unknown) => value === null || value === undefined || value === '' ? 'Unavailable' : String(value)

export default function RiskRanking({limit}: {limit?: number}) {
  const [kind, setKind] = useState<Kind>('Warehouse')
  const [rows, setRows] = useState<RiskWatchlistRow[] | null>(null)
  const [error, setError] = useState(false)
  const [level, setLevel] = useState('All')
  const [exposure, setExposure] = useState('All')
  const [place, setPlace] = useState('')
  const [sort, setSort] = useState('authoritative')
  const [reload, setReload] = useState(0)
  useEffect(() => {let active = true; setRows(null); setError(false); getCurrentRiskWatchlist(kind).then(result => {if (active) setRows(result)}).catch(() => {if (active) setError(true)}); return () => {active = false}}, [kind, reload])
  const visible = useMemo(() => {
    const filtered = (rows || []).filter(row => (level === 'All' || row.risk_level === level) && (exposure === 'All' || exposure === 'Operational exposure' && row.operational_event_count > 0 || exposure === 'No operational exposure' && row.operational_event_count === 0) && `${row.entity_name} ${row.location || ''}`.toLowerCase().includes(place.toLowerCase()))
    if (sort === 'score') return [...filtered].sort((a, b) => (b.risk_score ?? -1) - (a.risk_score ?? -1))
    if (sort === 'name') return [...filtered].sort((a, b) => a.entity_name.localeCompare(b.entity_name))
    return filtered
  }, [rows, level, exposure, place, sort])
  return <section className="card ranking-card"><div className="card-title"><div><p className="eyebrow">CURRENT · RULE-BASED RISK</p><h2>Evidence-backed ranking</h2><p className="muted">Default order is the authoritative API ranking. Frontend sorting is explicitly labeled.</p></div><span className="badge">{kind.toUpperCase()}</span></div><div className="ranking-controls"><label>Entity type<select value={kind} onChange={event => setKind(event.target.value as Kind)}><option>Warehouse</option><option>Product</option><option>Supplier</option></select></label><label>Risk level<select value={level} onChange={event => setLevel(event.target.value)}>{['All','Critical','High','Medium','Low','Unavailable'].map(item => <option key={item}>{item}</option>)}</select></label><label>Operational exposure<select value={exposure} onChange={event => setExposure(event.target.value)}><option>All</option><option>Operational exposure</option><option>No operational exposure</option></select></label><label>Location / entity<input value={place} onChange={event => setPlace(event.target.value)} placeholder="Filter by name or location"/></label><label>Order<select value={sort} onChange={event => setSort(event.target.value)}><option value="authoritative">Authoritative API ranking</option><option value="score">Derived score (local)</option><option value="name">Name (local)</option></select></label></div>{error ? <div className="empty-state"><p>Risk ranking is unavailable; no Low state has been inferred.</p><button className="secondary compact-btn" onClick={() => setReload(value => value + 1)}>Retry</button></div> : rows === null ? <p className="muted">Loading authoritative ranking…</p> : !visible.length ? <p className="muted">No rows match these filters. Clear filters to review the current ranking.</p> : <div className="ranking-scroll"><table className="ranking-table"><thead><tr><th scope="col">Rank</th><th scope="col">Entity / location</th><th scope="col">Risk</th><th scope="col">Derived score</th><th scope="col">Operational events</th><th scope="col">Top event / severity</th><th scope="col">Availability / reason</th><th scope="col">Updated</th></tr></thead><tbody>{visible.slice(0, limit || visible.length).map(row => <tr key={row.entity_id}><td>{rows.indexOf(row) + 1}</td><td><Link to={`/${kind.toLowerCase()}s/${encodeURIComponent(row.entity_id)}`}>{row.entity_name}</Link><small>{display(row.location)} · {row.entity_type}</small>{row.catalog_only && <em>Catalog reference · stock unverified</em>}</td><td><span className={`badge ${row.risk_level.toLowerCase()}`}>{row.risk_level}</span></td><td>{row.risk_score === null ? 'Unavailable' : row.risk_score.toFixed(3)}</td><td>{row.operational_event_count} <small>{row.active_signal_count} active signals</small></td><td>{row.top_event || 'No active operational event'}<small>{display(row.top_event_severity)}</small></td><td>{display(row.risk_availability)}<small>{row.reason}</small></td><td>{row.updated_at ? new Date(row.updated_at).toLocaleString() : 'Unavailable'}</td></tr>)}</tbody></table></div>}</section>
}
