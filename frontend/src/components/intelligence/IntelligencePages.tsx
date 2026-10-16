import {useEffect, useState} from 'react'
import {Link, useParams} from 'react-router-dom'
import {ArrowLeft, ArrowRight, Boxes, Factory, GitBranch, Search, Warehouse as WarehouseIcon} from 'lucide-react'
import {useCurrentSnapshot} from '../../hooks/useCurrentSnapshot'
import {getGraphEntity, getWarehouseCatalog} from '../../services/supplyBeaconApi'
import type {CatalogAssociation, Entity, GraphEntity} from '../../services/supplyBeaconApi'

const value = (input: unknown) => input === null || input === undefined || input === '' ? 'Unavailable' : String(input)
const score = (input: unknown) => typeof input === 'number' && Number.isFinite(input) ? input.toFixed(3) : 'Unavailable'
const id = (entity: Entity) => value(entity.node_id || entity.Node_ID)
const location = (entity: Entity) => [entity.city, entity.state].filter(part => part && part !== 'Unavailable').join(', ') || 'Location unavailable'

function State({error, retry}: {error?: boolean; retry?: () => void}) {
  return <section className="card empty-card"><h2>{error ? 'Current data unavailable' : 'Loading current graph…'}</h2><p className="muted">{error ? 'The current-data API could not be reached.' : 'Retrieving authoritative graph and risk evidence.'}</p>{error && <button className="primary compact-btn" onClick={retry}>Retry</button>}</section>
}

export function EntityListPage({type}: {type: 'Product' | 'Warehouse' | 'Supplier'}) {
  const data = useCurrentSnapshot()
  const [query, setQuery] = useState('')
  if (!data || data.error) return <State error={Boolean(data?.error)} retry={data?.retry}/>
  const entities = (type === 'Product' ? data.products : type === 'Warehouse' ? data.warehouses : data.suppliers)
    .filter(entity => `${entity.name} ${entity.node_id} ${entity.city || ''} ${entity.state || ''}`.toLowerCase().includes(query.toLowerCase()))
  return <><div className="page-heading"><div><p className="eyebrow">INTELLIGENCE · CURRENT GRAPH</p><h1>{type === 'Product' ? 'Products' : `${type}s`}</h1><p className="muted">{type === 'Warehouse' ? 'Operational inventory and curated catalog references are distinct relationships.' : type === 'Product' ? 'Operational Product nodes; catalog ProductModels are reference records, not stocked SKUs.' : 'Supplier exposure requires a typed path through Brand, Product, and STORED_AT.'}</p></div><label className="search-small"><Search/><span className="sr-only">Search {type.toLowerCase()}s</span><input value={query} onChange={event => setQuery(event.target.value)} placeholder={`Search ${type.toLowerCase()}s or locations…`}/></label></div>
    {type === 'Warehouse' && <p className="catalog-callout">Catalog reference means the warehouse and product model are source-verified; physical inventory is not verified.</p>}
    <div className="entity-grid">{entities.map(entity => <Link className="entity-card" to={`/${type.toLowerCase()}s/${encodeURIComponent(id(entity))}`} key={id(entity)}><div className="entity-icon">{type === 'Warehouse' ? <WarehouseIcon/> : type === 'Product' ? <Boxes/> : <Factory/>}</div><div className="entity-main"><h3>{value(entity.name)}</h3><span>{type === 'Warehouse' ? location(entity) : type === 'Product' ? `${value(entity.brand)} · ${value(entity.category)}` : 'Supplier → Brand → Product'}</span><small className={entity.catalog_only ? 'catalog-tag' : 'inventory-tag'}>{type === 'Warehouse' ? entity.catalog_only ? 'CATALOG REFERENCE · STOCK NOT VERIFIED' : 'VERIFIED GRAPH INVENTORY' : type === 'Product' ? 'OPERATIONAL PRODUCT' : 'TYPED GRAPH PATHS'}</small></div><div className="entity-score"><b>{value(entity.risk_level)}</b><small>{score(entity.risk_score)} derived score</small></div><ArrowRight/></Link>)}</div>
    {!entities.length && <section className="card empty-card"><h2>No matching {type.toLowerCase()}s</h2><p className="muted">Try another name or location. No risk or inventory status is inferred from an empty search.</p></section>}
  </>
}

function useGraph(nodeId: string) {
  const [graph, setGraph] = useState<GraphEntity | null>(null)
  const [error, setError] = useState(false)
  useEffect(() => {let active = true; setGraph(null); setError(false); if (nodeId) getGraphEntity(nodeId).then(result => {if (active) setGraph(result)}).catch(() => {if (active) setError(true)}); return () => {active = false}}, [nodeId])
  return {graph, error}
}

export function EntityDetailPage({type}: {type: 'Product' | 'Warehouse' | 'Supplier'}) {
  const {id: rawId} = useParams()
  const nodeId = decodeURIComponent(rawId || '')
  const data = useCurrentSnapshot()
  const {graph, error} = useGraph(nodeId)
  const [catalog, setCatalog] = useState<CatalogAssociation[] | null>(null)
  const [catalogError, setCatalogError] = useState(false)
  useEffect(() => {if (type !== 'Warehouse' || !nodeId) return; let active = true; setCatalog(null); setCatalogError(false); getWarehouseCatalog(nodeId).then(rows => {if (active) setCatalog(rows)}).catch(() => {if (active) setCatalogError(true)}); return () => {active = false}}, [type, nodeId])
  if (!data || !graph) return <State error={Boolean(data?.error || error)} retry={data?.retry}/>
  const entity = data.entities.find(item => id(item) === nodeId)
  const groups = new Map<string, GraphEntity['neighbors']>()
  for (const neighbor of graph.neighbors) groups.set(neighbor.relationship, [...(groups.get(neighbor.relationship) || []), neighbor])
  const stock = groups.get('STORED_AT') || []
  const catalogOnly = type === 'Warehouse' && Boolean(entity?.catalog_only)
  return <><Link className="back" to={`/${type.toLowerCase()}s`}><ArrowLeft/> Back to {type.toLowerCase()}s</Link><div className="detail-heading"><div><p className="eyebrow">{type.toUpperCase()} · CURRENT GRAPH</p><h1>{value(entity?.name || graph.name)}</h1><p className="muted">{type === 'Warehouse' ? location(entity || {}) : type === 'Product' ? `${value(entity?.brand)} · ${value(entity?.category)}` : 'Graph-backed supplier relationships'}</p></div><span className="badge">{value(entity?.risk_level)}</span></div>
    {catalogOnly && <section className="catalog-callout"><strong>Catalog reference — physical inventory not verified.</strong><p>ProductModel associations below do not create STORED_AT inventory or operational product impact.</p></section>}
    <div className="detail-grid"><section className="card"><p className="eyebrow">DERIVED CURRENT RISK</p><h2>{value(entity?.risk_level)}</h2><p>{score(entity?.risk_score)} Derived Risk Score · {value(entity?.risk_availability)}</p><p className="muted">{value(entity?.risk_reason)}</p></section><section className="card"><p className="eyebrow">RELATIONSHIP EVIDENCE</p>{type === 'Warehouse' ? <><div className="insight-row"><span>Verified graph inventory</span><b>{entity?.inventory_relationship_available ? `${stock.length} STORED_AT relationships` : 'Unavailable'}</b></div><div className="insight-row"><span>Catalog associations</span><b>{value(entity?.catalog_association_count)}</b></div><div className="insight-row"><span>Operational events / active signals</span><b>{value(entity?.operational_event_count)} / {value(entity?.active_signal_count)}</b></div><div className="insight-row"><span>Operator · evidence grade</span><b>{value(entity?.operator)} · {value(entity?.evidence_grade)}</b></div></> : <><div className="insight-row"><span>Typed graph relationships</span><b>{graph.neighbors.length}</b></div><div className="insight-row"><span>Operational event count</span><b>{value(entity?.connected_events)}</b></div></>}</section></div>
    <section className="card"><div className="card-title"><div><p className="eyebrow">GRAPH NEIGHBORS</p><h3>{type === 'Warehouse' && !catalogOnly ? 'Verified Graph Inventory & Relationships' : 'Typed Relationships'}</h3></div><span className="muted">{graph.neighbors.length} edges</span></div>{groups.size ? [...groups].map(([relationship, neighbors]) => <div className="relationship-group" key={relationship}><h4>{relationship} <small>{neighbors.length}</small></h4><div className="relationship-list">{neighbors.slice(0, 30).map(neighbor => <div className="relationship-row" key={`${relationship}-${neighbor.node_id}`}><span><strong>{neighbor.name}</strong><small>{neighbor.node_type} · {neighbor.node_id}</small></span><span>{relationship === 'CATALOG_ASSOCIATED_WITH' ? 'REFERENCE ONLY' : value(neighbor.provenance)}</span></div>)}</div>{neighbors.length > 30 && <p className="muted">Showing first 30 of {neighbors.length} relationships.</p>}</div>) : <p className="muted">No graph relationships are available for this entity.</p>}</section>
    {type === 'Warehouse' && <section className="card"><div className="card-title"><div><p className="eyebrow">CURATED REFERENCE LAYER</p><h3>Catalog Associations</h3></div><span className="badge">NOT VERIFIED INVENTORY</span></div>{catalogError ? <p className="muted">Catalog associations could not be loaded.</p> : catalog === null ? <p className="muted">Loading catalog associations…</p> : catalog.length ? <><p className="muted">{catalog.length} source-verified product model associations; none certify stock at this warehouse.</p><div className="relationship-list">{catalog.slice(0, 30).map(item => <div className="relationship-row" key={item.node_id}><span><strong>{item.product_model}</strong><small>{value(item.brand)} · {value(item.category)}</small></span><span>CATALOG REFERENCE</span></div>)}</div>{catalog.length > 30 && <p className="muted">Showing first 30 of {catalog.length} associations.</p>}</> : <p className="muted">No catalog associations are recorded for this warehouse.</p>}</section>}
  </>
}

const relations = ['STORED_AT', 'CATALOG_ASSOCIATED_WITH', 'LOCATED_IN', 'SUPPLIES', 'PRODUCES', 'AFFECTS']
export function GraphIntelligencePage() {
  const data = useCurrentSnapshot()
  const [nodeId, setNodeId] = useState('')
  const [type, setType] = useState('Warehouse')
  const [query, setQuery] = useState('')
  const {graph, error} = useGraph(nodeId)
  if (!data || data.error) return <State error={Boolean(data?.error)} retry={data?.retry}/>
  const candidates = data.entities.filter(entity => entity.node_type === type && `${entity.name} ${entity.node_id}`.toLowerCase().includes(query.toLowerCase())).slice(0, 25)
  return <><div className="page-heading"><div><p className="eyebrow">INTELLIGENCE · BOUNDED GRAPH</p><h1>Graph Intelligence</h1><p className="muted">Inspect real typed neighbors of a selected entity. No shortest-path inventory claims are inferred.</p></div><span className="badge">{value(data.summary.graph_nodes)} NODES · {value(data.summary.graph_edges)} EDGES</span></div><div className="graph-browser"><section className="card"><div className="card-title"><div><p className="eyebrow">SELECT ENTITY</p><h3>Browse a bounded graph neighborhood</h3></div><GitBranch/></div><div className="graph-controls"><label>Entity type<select value={type} onChange={event => {setType(event.target.value); setNodeId('')}}><option>Warehouse</option><option>Product</option><option>Supplier</option></select></label><label>Find entity<input value={query} onChange={event => setQuery(event.target.value)} placeholder="Name or node ID"/></label></div><div className="graph-picker">{candidates.map(entity => <button className={nodeId === id(entity) ? 'active' : ''} onClick={() => setNodeId(id(entity))} key={id(entity)}>{value(entity.name)}<small>{id(entity)}</small></button>)}</div>{!candidates.length && <p className="muted">No matching entities.</p>}</section><section className="card"><p className="eyebrow">SELECTED NODE</p>{!nodeId ? <p className="muted">Select an entity to inspect authoritative graph edges.</p> : error ? <p className="muted">Graph detail is unavailable for this entity.</p> : !graph ? <p className="muted">Loading graph neighbors…</p> : <><h2>{graph.name}</h2><p className="muted">{graph.node_type} · {graph.node_id} · {graph.neighbors.length} typed neighbors</p><div className="relationship-list">{graph.neighbors.slice(0, 40).map(item => <div className="relationship-row" key={`${item.relationship}-${item.node_id}`}><span><strong>{item.name}</strong><small>{item.node_type} · {item.node_id}</small></span><span className={item.relationship === 'CATALOG_ASSOCIATED_WITH' ? 'catalog-tag' : 'inventory-tag'}>{item.relationship}</span></div>)}</div>{graph.neighbors.length > 40 && <p className="muted">Showing first 40 of {graph.neighbors.length} neighbors.</p>}</>}</section></div><section className="card"><p className="eyebrow">RELATIONSHIP LEGEND</p><div className="relationship-legend">{relations.map(relation => <span key={relation} className={relation === 'CATALOG_ASSOCIATED_WITH' ? 'catalog-tag' : 'inventory-tag'}>{relation}</span>)}</div><p className="muted">STORED_AT is an operational product–warehouse inventory relationship. CATALOG_ASSOCIATED_WITH is a reference association only. AFFECTS requires validated event and graph resolution.</p></section></>
}
