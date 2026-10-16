import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent, ReactNode } from 'react'
import { AlertTriangle, ArrowRight, ChevronDown, Copy, GitBranch, RefreshCw, Search, Trash2 } from 'lucide-react'
import { AgenticResponse, Entity, Event, askSupplyBeaconAgentic } from '../../services/supplyBeaconApi'

type AskMessage =
  | { id:string; role:'user'; text:string; createdAt:number }
  | { id:string; role:'assistant'; response:AgenticResponse; createdAt:number }
  | { id:string; role:'error'; query:string; message:string; createdAt:number }

const suggestions = [
  'Which warehouses are currently at highest risk?',
  'Why is Delhi Warehouse risky and what happens if it closes?',
  'Which events are currently affecting warehouses?',
  'What products are associated with WH_REAL_066? Are they physically stored there?',
  'Is the current risk score a probability?',
  'What does the historical DataCo model represent?',
]

const readable = (value:unknown, fallback='Unavailable') => typeof value === 'string' && value.trim() ? value : typeof value === 'number' && Number.isFinite(value) ? String(value) : fallback
const numberValue = (value:unknown) => typeof value === 'number' && Number.isFinite(value) ? value : typeof value === 'string' && Number.isFinite(Number(value)) ? Number(value) : null
const intentLabel = (value?:string) => value ? value.split('_').map(word => word.charAt(0) + word.slice(1).toLowerCase()).join(' ') : 'Unsupported / Unknown'
const riskLabel = (value:unknown) => ['Low','Medium','High','Critical'].includes(String(value)) ? String(value) : 'Unavailable'
const formatTimestamp = (value:unknown) => { if (typeof value !== 'string') return 'Unavailable'; const date = new Date(value); return Number.isNaN(date.getTime()) ? 'Unavailable' : date.toLocaleString() }
const titleCase = (value:unknown) => readable(value).replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, char => char.toUpperCase())
const record = (value:unknown):Record<string, unknown> => value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
const metric = (items:unknown[], predicate:string) => items.map(record).find(item => item.predicate === predicate)?.value
const provenanceLabel = (value:unknown) => titleCase(value).replace('Rule Based', 'Rule-Based')

function Disclosure({title,children,open=false}:{title:string;children:ReactNode;open?:boolean}) {
  return <details className="ask-disclosure" open={open}><summary>{title}<ChevronDown/></summary><div>{children}</div></details>
}
function ProvenanceBadge({value}:{value:unknown}) {
  const label = provenanceLabel(value)
  const descriptions:Record<string,string> = {
    Simulated:'Generated or mapped project data rather than directly observed source data.',
    Derived:'Computed from validated relationships or business rules.',
    'Derived / Rule-Based':'Computed using SupplyBeacon rules, not a learned probability.',
    'Model Output':'Produced by the existing risk artifact or historical model.',
    'External Api':'Retained external API evidence.',
    'Historical Model':'Metric from the historical DataCo supervised model.',
  }
  return <span className="ask-provenance" title={descriptions[label] || ''}>{label}</span>
}
function RiskCard({evidence}:{evidence:unknown[]}) {
  if (!evidence.length) return null
  const items = evidence.map(record)
  const first = items[0]
  const score = metric(items, 'Risk_Score') ?? metric(items, 'Final_Scenario_Risk_Score')
  const fields = [
    ['Entity', first.subject || first.name],
    ['Risk Level', riskLabel(metric(items, 'Risk_Level'))],
    ['Scenario Risk Score', numberValue(score)?.toFixed(3)],
    ['Connected Events', metric(items, 'Connected_Events')],
    ['Risk Reason', metric(items, 'Risk_Reason')],
    ['Structural Baseline', metric(items, 'Structural_Baseline')],
    ['Weather Component', metric(items, 'Weather_Event_Component')],
    ['News Component', metric(items, 'News_Event_Component')],
    ['Business Impact', metric(items, 'Business_Impact_Component')],
    ['Event Exposure', metric(items, 'Event_Exposure_Component')],
  ].filter(([,value]) => value !== undefined && value !== null && value !== '')
  return <section className="ask-business-card"><div className="ask-section-heading"><h3>Current Risk</h3><ProvenanceBadge value={first.provenance}/></div><div className="ask-fact-grid">{fields.map(([label,value]) => <div key={String(label)}><small>{String(label)}</small><strong>{readable(value)}</strong></div>)}</div></section>
}
function RankingCard({entities}:{entities:Entity[]}) {
  return <section className="ask-business-card"><div className="ask-section-heading"><h3>Ranked Current Results</h3><span>{entities.length} items</span></div><div className="ask-ranking-list">{entities.map((entity,index)=><article className="ask-ranking-row" key={entity.node_id || entity.name || index}><span className="ask-rank">{index + 1}</span><div><strong>{readable(entity.name)}</strong><small>{titleCase(entity.node_type)}</small></div><div><small>Risk</small><strong>{riskLabel(entity.risk_level)}</strong></div><div><small>Scenario score</small><strong>{numberValue(entity.risk_score)?.toFixed(3) ?? 'Unavailable'}</strong></div><div><small>Connected events</small><strong>{readable(entity.connected_events)}</strong></div><div><ProvenanceBadge value={entity.provenance || entity.source_artifact || 'MODEL_OUTPUT'}/></div>{Boolean(entity.risk_reason) && <p>{readable(entity.risk_reason)}</p>}</article>)}</div></section>
}
function EventCard({event}:{event:Event}) {
  const item = record(event)
  const title = event.title || item.Title || item.title || item.name
  const eventType = event.event_type || item.Event_Type || item.event_type
  const domain = event.event_domain || item.event_domain
  const source = event.source_channel || item.source_channel
  const location = event.location || item.Location || item.location
  const severity = event.severity || item.Severity || event.impact_level || item.Impact_Level
  const operational = event.operationally_relevant ?? item.Operationally_Relevant
  const effect = event.possible_supply_chain_effect || item.Possible_Supply_Chain_Effect
  const status = String(event.event_status || item.event_status || 'UNKNOWN').toUpperCase()
  const statusLabel = status === 'ACTIVE' ? 'CURRENT EXTERNAL EVIDENCE' : status === 'UPCOMING' ? 'UPCOMING EXTERNAL EVIDENCE' : status === 'EXPIRED' ? 'RECENT / HISTORICAL EXTERNAL EVIDENCE' : 'EXTERNAL EVIDENCE · TEMPORAL STATUS UNKNOWN'
  const statusText = status.charAt(0) + status.slice(1).toLowerCase()
  const eventStart = event.event_start || item.event_start
  const eventEnd = event.event_end || item.event_end
  const temporalBasis = event.temporal_basis || item.temporal_basis
  return <article className="ask-event-card"><div className="ask-event-title"><strong>{readable(title, 'Operational event')}</strong><ProvenanceBadge value={event.provenance || item.provenance || (source ? 'EXTERNAL_API' : undefined)}/></div><div className="ask-event-status"><small>{statusLabel}</small><strong>Event Status: {statusText}</strong></div><div className="ask-fact-grid ask-event-facts"><div><small>Event class / type</small><strong>{titleCase(event.event_class || item.Event_Class || eventType)}</strong></div><div><small>Domain</small><strong>{titleCase(domain)}</strong></div><div><small>Source channel</small><strong>{titleCase(source)}</strong></div><div><small>Location</small><strong>{readable(location)}</strong></div><div><small>Severity</small><strong>{riskLabel(severity)}</strong></div><div><small>Operational status</small><strong>{operational === true ? 'Operationally Relevant' : 'Context / Monitoring'}</strong></div><div><small>Timestamp</small><strong>{formatTimestamp(item.Date || item.date)}</strong></div><div><small>Forecast valid</small><strong>{eventStart ? `${formatTimestamp(eventStart)}${eventEnd ? ` - ${formatTimestamp(eventEnd)}` : ''}` : 'Unavailable'}</strong></div><div><small>Temporal provenance</small><strong>{readable(temporalBasis)}</strong></div></div>{Boolean(effect) && <p className="muted">Possible effect: {readable(effect)}</p>}</article>
}
function GraphPathCard({path}:{path:Record<string,unknown>}) {
  const nodes = Array.isArray(path.nodes) ? path.nodes : []
  const edges = Array.isArray(path.edges) ? path.edges.map(record) : []
  if (!nodes.length) return null
  return <article className="ask-path-card"><div className="ask-path-flow">{nodes.map((node,index)=><span key={`${String(node)}-${index}`}><b>{String(node).replace(/^(PRODUCT|WAREHOUSE|LOCATION|CURRENT_EVENT|SUPPLIER|BRAND):/, '').replace(/_/g, ' ')}</b>{index < nodes.length - 1 && <em>{readable(edges[index]?.relationship, 'CONNECTED_TO')}</em>}</span>)}</div>{edges.some(edge => edge.provenance || edge.exposure_scope) && <div className="ask-path-meta">{edges.map((edge,index)=><span key={index}><ProvenanceBadge value={edge.provenance}/>{Boolean(edge.exposure_scope) && ` ${String(edge.exposure_scope).replace(/_/g, ' ')}`}</span>)}</div>}</article>
}
function HistoricalCard({historical}:{historical:Record<string, unknown>}) {
  const metrics = Array.isArray(historical.test_metrics) ? historical.test_metrics.map(record) : []
  const best = metrics.find(item => item.model === historical.best_model) || metrics.find(item => item.model === 'hist_gradient_boosting')
  if (!metrics.length && !historical.best_model) return null
  const modelLabel = readable(historical.best_model).replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase())
  return <section className="ask-business-card"><div className="ask-section-heading"><h3>Historical Shipping Intelligence</h3><ProvenanceBadge value="HISTORICAL_MODEL"/></div><div className="ask-model-meta"><span><small>Model</small><strong>{modelLabel}</strong></span><span><small>Dataset</small><strong>DataCo Supply Chain</strong></span><span><small>Split</small><strong>Test</strong></span></div>{best !== undefined && <div className="ask-metric-row">{[['MAE',best.mae],['RMSE',best.rmse],['R²',best.r2]].map(([label,value])=><div key={String(label)}><small>{String(label)}</small><strong>{numberValue(value)?.toFixed(3) ?? 'Unavailable'}</strong></div>)}</div>}<p className="muted">Historical model performance does not represent today&apos;s scenario-risk score.</p></section>
}
function FreshnessCard({response}:{response:AgenticResponse}) {
  const metadata = response.current_metadata || {}
  const status = response.current_data_fresh === true ? 'Fresh' : response.current_data_fresh === false ? 'Stale' : 'Unavailable'
  return <section className={`ask-business-card ask-freshness ${status.toLowerCase()}`}><div className="ask-section-heading"><h3>Freshness & Trust</h3><strong>{status}</strong></div>{status === 'Stale' && <p className="ask-warning">Current external-signal evidence may be stale.</p>}{status === 'Unavailable' && <p className="muted">Freshness unavailable.</p>}<div className="ask-fact-grid"><div><small>Weather fetched</small><strong>{formatTimestamp(metadata.weather_fetch_timestamp)}</strong></div><div><small>News fetched</small><strong>{formatTimestamp(metadata.news_fetch_timestamp)}</strong></div><div><small>Pipeline run</small><strong>{formatTimestamp(metadata.pipeline_run_timestamp || metadata.run_timestamp)}</strong></div><div><small>Weather age</small><strong>{numberValue(metadata.weather_age_hours)?.toFixed(1) ?? 'Unavailable'} hours</strong></div><div><small>News age</small><strong>{numberValue(metadata.news_age_hours)?.toFixed(1) ?? 'Unavailable'} hours</strong></div></div></section>
}
function AssistantCard({response,onCopy}:{response:AgenticResponse;onCopy:(text:string)=>void}) {
  const fallback = response.fallback_used
  const plan = response.query_plan
  const ranking = ['PRODUCT_RISK','SUPPLIER_RISK'].includes(plan.primary_intent) && plan.entity_mentions.length === 0 && response.entities.length > 0
  return <div className="ask-assistant-block"><section className={`card ask-answer ${fallback ? 'ask-fallback' : ''}`}><div className="ask-answer-top"><p className="eyebrow">{fallback ? 'INSUFFICIENT GROUNDED EVIDENCE' : 'GROUNDED ANSWER'}</p><button className="icon-btn" onClick={()=>onCopy(response.answer)} aria-label="Copy answer" title="Copy answer"><Copy/></button></div><h2>{readable(response.answer)}</h2><div className="ask-meta"><span><b>Intent</b>{intentLabel(plan.primary_intent)}</span>{plan.secondary_intents.length > 0 && <span><b>Also</b>{plan.secondary_intents.map(intentLabel).join(' + ')}</span>}<span><b>Freshness</b>{response.current_data_fresh === true ? 'Fresh' : response.current_data_fresh === false ? 'Stale' : 'Unavailable'}</span><span><b>Evidence</b>{response.evidence_count} items</span><span><b>Generation</b>{response.llm_used ? 'LLM-assisted grounded explanation' : 'Deterministic evidence response'}</span></div></section>{fallback && Boolean(response.limitations?.length) && <section className="ask-business-card ask-limitations"><h3>Why this answer is limited</h3><ul>{response.limitations?.map(item=><li key={item}>{item}</li>)}</ul></section>}{ranking ? <RankingCard entities={response.entities}/>: response.entities.length > 0 && <section className="ask-business-card"><div className="ask-section-heading"><h3>Resolved Entities</h3><span>{response.entities.length}</span></div><div className="ask-entity-list">{response.entities.map(entity=><div className="ask-entity-card" key={entity.node_id || entity.name}><strong>{readable(entity.name)}</strong><span>{titleCase(entity.node_type)}</span>{entity.risk_level && <span>Risk: {riskLabel(entity.risk_level)}</span>}</div>)}</div></section>}{!ranking && <RiskCard evidence={response.risk_evidence}/>} {response.events.length > 0 && <section className="ask-business-card"><div className="ask-section-heading"><h3>Operational Events</h3><span>{response.events.length}</span></div>{response.events.map((event,index)=><EventCard event={event} key={String(event.event_id ?? event.node_id ?? index)}/>)}</section>}{response.graph_paths.length > 0 && <section className="ask-business-card"><div className="ask-section-heading"><h3>Graph Evidence</h3><GitBranch/></div>{response.graph_paths.map((path,index)=><GraphPathCard path={record(path)} key={index}/>)}</section>}{response.delivery_risk && <section className="ask-business-card"><div className="ask-section-heading"><h3>Delivery Scenario</h3><ProvenanceBadge value="DERIVED / RULE_BASED"/></div><div className="ask-fact-grid">{Object.entries(response.delivery_risk).filter(([,value])=>value !== null && value !== undefined && typeof value !== 'object').map(([key,value])=><div key={key}><small>{titleCase(key)}</small><strong>{String(value)}</strong></div>)}</div><p className="muted">Scenario estimate — not a guaranteed delivery delay.</p></section>}{response.historical_evidence && Object.keys(response.historical_evidence).length > 0 && <HistoricalCard historical={response.historical_evidence}/>}<FreshnessCard response={response}/><Disclosure title="Developer Details"><p className="muted">Workflow: {response.workflow_trace.map(titleCase).join(' → ')}</p><pre>{JSON.stringify({evidence_used:response.evidence_used,timings:response.timings,response},null,2)}</pre></Disclosure></div>
}
function GroundedDetails({response}:{response:AgenticResponse}) {
  const scenario = record(response.scenario_result)
  const catalog = record(response.catalog_context)
  const candidates = Array.isArray(scenario.alternative_warehouse_candidates) ? scenario.alternative_warehouse_candidates.map(record) : []
  const sources = [...new Set((response.evidence || []).map(item => String(item.source_type || '')).filter(Boolean))]
  return <>
    {Object.keys(scenario).length > 0 && <section className="ask-business-card"><h3>Modeled warehouse scenario — not a forecast</h3><p>Verified inventory relation: {scenario.inventory_relationship_available === true ? 'Available' : 'Unavailable'}</p>{scenario.inventory_relationship_available === false ? <p className="ask-warning">Operational product impact cannot be established from verified inventory relationships. Catalog associations are not physical stock.</p> : <p>Verified affected products: {String(scenario.affected_product_count ?? 'Unavailable')} · Suppliers linked to those products: {String(scenario.affected_supplier_count ?? 'Unavailable')}</p>}<p>Alternative warehouse graph candidates: {candidates.length}. These are not confirmed logistics or capacity options.</p></section>}
    {Object.keys(catalog).length > 0 && <section className="ask-business-card"><h3>Catalog / reference context</h3><p>{String(catalog.catalog_association_count ?? 0)} ProductModel associations. Physical inventory is not verified by these associations.</p><p>Verified STORED_AT products: {Array.isArray(catalog.verified_stored_product_ids) ? catalog.verified_stored_product_ids.length : 'Unavailable'}</p></section>}
    {sources.length > 0 && <section className="ask-business-card"><h3>Evidence sources</h3><p>{sources.join(' · ')}</p></section>}
    {response.warnings.length > 0 && <section className="ask-business-card ask-limitations"><h3>Warnings</h3><ul>{response.warnings.map(item => <li key={item}>{item}</li>)}</ul></section>}
    {!response.fallback_used && Boolean(response.limitations?.length) && <section className="ask-business-card ask-limitations"><h3>Limitations</h3><ul>{response.limitations?.map(item => <li key={item}>{item}</li>)}</ul></section>}
  </>
}
function AskMessageView({message,onCopy,onRetry}:{message:AskMessage;onCopy:(text:string)=>void;onRetry:(text:string)=>void}) {
  if (message.role === 'user') return <div className="ask-user-message"><span>{message.text}</span></div>
  if (message.role === 'error') return <section className="card ask-error" role="alert"><AlertTriangle/><div><strong>ASK SUPPLYBEACON UNAVAILABLE</strong><p>{message.message}</p><button className="secondary compact-btn" onClick={()=>onRetry(message.query)}><RefreshCw/> Retry</button><button className="secondary compact-btn" onClick={()=>onCopy(message.query)}><ClipboardIcon/> Copy question</button></div></section>
  return <><AssistantCard response={message.response} onCopy={onCopy}/><GroundedDetails response={message.response}/></>
}
function ClipboardIcon() { return <span aria-hidden="true"><Copy/></span> }

export function AskPage() {
  const [query,setQuery] = useState('')
  const [messages,setMessages] = useState<AskMessage[]>([])
  const [loading,setLoading] = useState(false)
  const [pendingQuery,setPendingQuery] = useState('')
  const endRef = useRef<HTMLDivElement>(null)
  const submit = async (value=query) => {
    const text = value.trim()
    if (!text || loading) return
    const now = Date.now()
    setMessages(previous => [...previous, {id:`user-${now}`,role:'user',text,createdAt:now}])
    setPendingQuery(text)
    setLoading(true)
    try {
      const response = await askSupplyBeaconAgentic(text)
      setMessages(previous => [...previous, {id:`assistant-${Date.now()}`,role:'assistant',response,createdAt:Date.now()}])
      setQuery('')
    } catch {
      setMessages(previous => [...previous, {id:`error-${Date.now()}`,role:'error',query:text,message:'Could not reach the SupplyBeacon intelligence service.',createdAt:Date.now()}])
    } finally { setPendingQuery(''); setLoading(false) }
  }
  useEffect(() => { if (messages.length) endRef.current?.scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',block:'nearest'}) }, [messages])
  const copy = (text:string) => { void navigator.clipboard?.writeText(text) }
  const onKey = (event:KeyboardEvent<HTMLTextAreaElement>) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void submit() } }
  return <><div className="page-heading ask-heading"><div><p className="eyebrow">AI & ANALYTICS · AGENTIC GRAPHRAG</p><h1>Ask SupplyBeacon</h1><p className="muted">Ask questions about current supply-chain risk, operational events, delivery scenarios and historical shipping performance.</p></div>{messages.length > 0 && <button className="secondary compact-btn" onClick={()=>setMessages([])} disabled={loading}><Trash2/> Clear conversation</button>}</div><section className="ask-card ask-composer"><div className="ask-icon"><Search/></div><textarea aria-label="Ask SupplyBeacon a question" value={query} onChange={event=>setQuery(event.target.value)} onKeyDown={onKey} placeholder="Ask about current risk, events or historical performance..." disabled={loading}/><button className="primary" disabled={loading || !query.trim()} onClick={()=>void submit()}>{loading ? 'Retrieving…' : 'Ask SupplyBeacon'} <ArrowRight/></button></section>{loading && <div className="ask-loading" aria-live="polite"><RefreshCw className="ask-spin"/><span>Retrieving and validating SupplyBeacon evidence…</span></div>}{messages.length === 0 && <div className="suggestions">{suggestions.map(item=><button key={item} onClick={()=>setQuery(item)}>{item}<ArrowRight/></button>)}</div>}<div className="ask-conversation">{messages.map(message=><div key={message.id} className="ask-message"><AskMessageView message={message} onCopy={copy} onRetry={value=>{setQuery(value);void submit(value)}}/></div>)}{pendingQuery && <span className="sr-only">Processing {pendingQuery}</span>}<div ref={endRef}/></div></>
}
