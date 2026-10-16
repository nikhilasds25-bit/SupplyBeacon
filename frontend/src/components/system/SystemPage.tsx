import {useEffect, useState} from 'react'
import {Database, ShieldCheck} from 'lucide-react'
import {useCurrentSnapshot} from '../../hooks/useCurrentSnapshot'

const sections = [
  ['PRODUCT CATALOG', 'Original catalog and product attributes describe products; a catalog association is not a stocked SKU.'],
  ['OPERATIONAL GRAPH', 'Supplier → SUPPLIES → Brand → PRODUCES → Product and Product → STORED_AT → Warehouse are typed graph relationships.'],
  ['CURATED REFERENCE DATA', 'Manually curated, source-verified warehouses and product models add a reference layer. Their catalog associations do not verify physical stock.'],
  ['LIVE EXTERNAL DATA', 'Validated weather and news observations feed event extraction. Geographic and temporal checks govern operational eligibility.'],
  ['CURRENT RISK', 'Deterministic, rule-based scenario scores reflect current eligible evidence and graph paths; they are not calibrated disruption probabilities.'],
  ['HISTORICAL ML', 'The DataCo model predicts historical shipping days. It is separate from current operational risk.'],
  ['AI LAYER', 'GraphRAG retrieves structured evidence; LangGraph orchestrates; an optional LLM explains. Authoritative values come from deterministic services, not the LLM.'],
] as const
const formatTime = (input: unknown) => typeof input === 'string' && !Number.isNaN(Date.parse(input)) ? new Date(input).toLocaleString() : 'Unavailable'
const freshness = (input: unknown) => input === true ? 'Stale' : input === false ? 'Fresh' : 'Unavailable'

export default function SystemPage() {
  const data = useCurrentSnapshot()
  const [theme, setTheme] = useState(() => localStorage.getItem('supplybeacon-theme') || 'dark')
  useEffect(() => {document.documentElement.dataset.theme = theme; localStorage.setItem('supplybeacon-theme', theme); window.dispatchEvent(new Event('supplybeacon-theme-changed'))}, [theme])
  if (!data) return <div className="loading">Loading current provenance…</div>
  if (data.error) return <section className="card empty-card"><h2>Current metadata unavailable</h2><p className="muted">The API could not be reached.</p><button className="primary compact-btn" onClick={data.retry}>Retry</button></section>
  const metadata = data.metadata
  return <><div className="page-heading"><div><p className="eyebrow">SYSTEM</p><h1>System & Provenance</h1><p className="muted">What the product knows, how it knows it, and where evidence stops.</p></div><span className="badge">EVIDENCE LINEAGE</span></div><div className="grid-2"><section className="card"><p className="eyebrow">CURRENT SNAPSHOT</p><h2>Freshness & coverage</h2><div className="insight-row"><span>Weather</span><b>{freshness(metadata.weather_stale ?? data.liveSituation.freshness.weather_stale)}</b></div><div className="insight-row"><span>News</span><b>{freshness(metadata.news_stale ?? data.liveSituation.freshness.news_stale)}</b></div><div className="insight-row"><span>Weather observations</span><b>{data.liveSituation.coverage.represented} / {data.liveSituation.coverage.required} locations</b></div><div className="insight-row"><span>Weather fetched</span><b>{formatTime(metadata.weather_fetch_timestamp)}</b></div><div className="insight-row"><span>News fetched</span><b>{formatTime(metadata.news_fetch_timestamp)}</b></div><div className="insight-row"><span>Pipeline run</span><b>{formatTime(metadata.pipeline_run_timestamp)}</b></div></section><section className="card"><p className="eyebrow">WORKSPACE</p><h2>Display settings</h2><p className="muted">This is a demo workspace with local demo sign-in, not production authentication.</p><div className="theme-buttons"><button aria-pressed={theme === 'dark'} className={theme === 'dark' ? 'primary' : 'secondary'} onClick={() => setTheme('dark')}>Dark theme</button><button aria-pressed={theme === 'light'} className={theme === 'light' ? 'primary' : 'secondary'} onClick={() => setTheme('light')}>Light theme</button></div><p className="muted">Changing the theme does not change data or risk semantics.</p></section></div><div className="provenance-cards">{sections.map(([heading, description]) => <section className="card" key={heading}><span className="provenance-icon">{heading === 'CURRENT RISK' || heading === 'AI LAYER' ? <ShieldCheck/> : <Database/>}</span><p className="eyebrow">{heading}</p><p>{description}</p></section>)}</div></>
}
