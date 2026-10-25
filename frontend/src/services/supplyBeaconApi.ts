export type RiskLevel = 'Low'|'Medium'|'High'|'Critical'
export type Entity = {
  node_id?: string
  Node_ID?: string
  name?: string
  node_type?: string
  risk_level?: RiskLevel|string
  risk_score?: number|null
  connected_events?: number
  [key:string]: unknown
}
export type Event = { event_id?: string|number; title?: string; event_type?: string; event_domain?: string; source_channel?: string; location?: string; severity?: RiskLevel|string; risk_score?: number; impact_type?: string; impact_level?: string; possible_supply_chain_effect?: string; operationally_relevant?: boolean; event_class?: string; event_start?: string|null; event_end?: string|null; event_status?: 'ACTIVE'|'UPCOMING'|'EXPIRED'|'UNKNOWN'|string; temporal_confidence?: string; temporal_basis?: string; [key:string]: unknown }
export type CurrentSummary = { operational_events:number; affected_warehouses:number; risky_products:number; risky_suppliers:number; high_critical_nodes:number; graph_nodes:number; graph_edges:number; overall_risk_level:string; risk_counts?:Record<string, Record<string, number>> }
export type LiveSignal = {event_id?:string|number;title:string;event_type:string;category:string;source_channel:string;location:string;timestamp?:string;event_status:string;severity:string;event_class:string;operational_eligible:boolean;operational_reason:string;temporal_basis:string;event_start?:string|null;event_end?:string|null;graph_resolution:{resolved:boolean;warehouse_ids:string[];location_ids:string[]};risk_contribution:string}
export type WeatherRecord = {city:string;provider:string;record_kind:'OBSERVATION'|'DAILY_FORECAST'|'ALERT'|string;condition:string;temperature_c?:number|null;rainfall_mm:Record<string,number>;wind:Record<string,number>;timestamp?:string|null;forecast_date?:string|null;active_signal:boolean;operational_impact:boolean;operational_significance:string}
export type LiveSituation = {generated_at:string;last_refreshed?:string;freshness:Record<string,unknown>;summary:{candidate_events:number;active_signals:number;active_weather_signals:number;recent_news_signals:number;operational_events:number;graph_resolved_events:number;affected_warehouses:number;network_risk:string};signals:LiveSignal[];active_signals:LiveSignal[];weather:Array<{city:string;available:boolean;observations:WeatherRecord[];forecasts:WeatherRecord[];alerts:WeatherRecord[]}>;coverage:{required:number;represented:number;missing:string[];locations:string[]};news:{relevant_candidates:LiveSignal[];filtered_non_operational:LiveSignal[]};why_risk:{headline:string;explanation:string;pipeline:Array<{label:string;count?:number;value?:string}>}}
export type Snapshot = { summary:CurrentSummary; liveSituation:LiveSituation; entities:Entity[]; products:Entity[]; warehouses:Entity[]; suppliers:Entity[]; events:Event[]; metadata:Record<string,unknown>; graph_paths?: unknown[]; provenance:'DERIVED / RULE_BASED'|'DEMO FALLBACK'; error?:string; retry?:()=>void }
export type ImpactTrace = { event:Event; graph_event_node_id:string; warehouses:Array<{node_id:string;name:string;product_ids:string[];location_ids?:string[]}>; product_ids:string[]; supplier_ids:string[]; product_paths?:unknown[]; supplier_paths?:Array<{supplier_id:string;brand_id:string;product_id:string;warehouse_id:string;location_ids:string[];event_id:string}>; risk_levels:Record<string,number>; product_risk_levels?:Record<string,number>; network_risk_levels?:Record<string,number>; provenance:string }
export type RiskWatchlistRow = {entity_id:string;entity_name:string;entity_type:string;risk_score:number|null;risk_level:string;risk_availability:string;operational_event_count:number;active_signal_count:number;top_event:string|null;top_event_severity:string|null;location:string|null;inventory_relationship_available:boolean|null;catalog_only:boolean|null;risk_components:Record<string,number|null>;reason:string;updated_at:string|null}
export type GraphNeighbor = {node_id:string;node_type:string;name:string;relationship:string;provenance:string;exposure_scope?:string;exposure_location?:string;resolution_provenance?:string}
export type GraphEntity = {node_id:string;node_type:string;name:string;attributes:Record<string,unknown>;neighbors:GraphNeighbor[]}
export type CatalogAssociation = {node_id:string;product_model:string;brand:string;category:string;subcategory:string;product_source_url?:string;catalog_as_of?:string;association_basis?:string;inventory_verified:false;operational_inventory_relation:false;provenance:string}
export type QueryPlan = {
  primary_intent:string
  secondary_intents:string[]
  entity_mentions:string[]
  resolved_entity_ids:string[]
  requested_entity_types:string[]
  needs_graph:boolean
  needs_risk:boolean
  needs_events:boolean
  needs_delivery:boolean
  needs_historical:boolean
  needs_freshness:boolean
  needs_weather?:boolean
  operational_events_only?:boolean
}
export type GraphPath = {nodes?:string[];edges?:Array<Record<string,unknown>>}
export type DeliveryRisk = Record<string,unknown> & {status?:string;message?:string}
export type AgenticTimings = Record<string,number>
export type StructuredAnswer = {
  answer:string
  primary_intent:string
  secondary_intents:string[]
  entities:Entity[]
  risk_level:string|null
  evidence_used:string[]
  warnings:string[]
  provenance_notes:string[]
  current_data_fresh:boolean|null
  llm_used:boolean
  fallback_used:boolean
}
export type AgenticResponse = {
  query:string
  answer:string
  query_plan:QueryPlan
  entities:Entity[]
  informational_entities:Entity[]
  graph_paths:GraphPath[]
  events:Event[]
  weather?:Record<string,unknown>[]
  risk_evidence:unknown[]
  delivery_risk?:DeliveryRisk|null
  historical_evidence?:Record<string,unknown>
  current_metadata?:Record<string,unknown>
  evidence_count:number
  evidence_used:string[]
  warnings:string[]
  current_data_fresh:boolean|null
  llm_used:boolean
  fallback_used:boolean
  workflow_trace:string[]
  timings:AgenticTimings
  structured_answer?:StructuredAnswer
  limitations?:string[]
  query_intent?:string
  evidence?:Array<Record<string,unknown>>
  scenario_result?:Record<string,unknown>|null
  catalog_context?:Record<string,unknown>|null
}
// Development uses Vite's same-origin proxy; production must configure the
// backend origin so API requests cannot fall through to the frontend SPA.
const configuredBase = String(import.meta.env.VITE_API_BASE_URL || '').trim()
const base = (configuredBase || (import.meta.env.DEV ? '/api' : '')).replace(/\/+$/, '')
const DEFAULT_REQUEST_TIMEOUT_MS = 20_000
const LONG_REQUEST_TIMEOUT_MS = 60_000

async function request<T>(path:string, init?:RequestInit, timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS):Promise<T> {
  if (import.meta.env.PROD && !configuredBase) {
    throw new Error('VITE_API_BASE_URL must be set to the deployed backend origin.')
  }
  const url = `${base}${path}`
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(url, {...init, signal: controller.signal})
    if (!response.ok) {
      const detail = await response.text().catch(() => '')
      throw new Error(`SupplyBeacon API request failed at ${url}: ${response.status}${detail ? ` ${detail}` : ''}`)
    }
    return await response.json() as T
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error(`SupplyBeacon API request timed out after ${Math.round(timeoutMs / 1000)} seconds at ${url}. The backend may be overloaded or the endpoint may be stalled.`)
    }
    if (error instanceof Error && error.message.startsWith('SupplyBeacon API request failed at ')) {
      throw error
    }
    throw new Error(`SupplyBeacon API could not be reached at ${url}: ${(error as Error).message}`)
  } finally {
    window.clearTimeout(timeoutId)
  }
}
export function getCurrentSummary(){return request<CurrentSummary>('/current/summary')}
export function getCurrentEvents(){return request<Event[]>('/current/events?operational_only=true')}
export function getProducts(){return request<Entity[]>('/current/products')}
export function getProduct(nodeId:string){return request<Entity>(`/current/products/${encodeURIComponent(nodeId)}`)}
export function getWarehouses(){return request<Entity[]>('/current/warehouses')}
export function getWarehouse(nodeId:string){return request<Entity>(`/current/warehouses/${encodeURIComponent(nodeId)}`)}
export function getWarehouseCatalog(nodeId:string){return request<CatalogAssociation[]>(`/current/warehouse-catalog/${encodeURIComponent(nodeId)}`)}
export function getSuppliers(){return request<Entity[]>('/current/suppliers')}
export function getCurrentMetadata(){return request<Record<string,unknown>>('/current/metadata')}
export function getCurrentWeather(){return request<Record<string,unknown>>('/current/weather')}
export function getCurrentImpact(){return request<ImpactTrace[]>('/current/impact')}
export function getCurrentRiskWatchlist(entityType:'Warehouse'|'Product'|'Supplier'='Warehouse'){return request<RiskWatchlistRow[]>(`/current/risk-watchlist?entity_type=${entityType}`)}
export function getLiveSituation(){return request<LiveSituation>('/current/live-situation')}
export type ScenarioResponse = { scenario_id:string; generated_at:string; scenario:Record<string,unknown>; current_state:Record<string,unknown>; scenario_state:Record<string,unknown>; delta:{newly_exposed_products:string[];newly_exposed_suppliers:string[];overlap_products:string[];overlap_suppliers:string[]}; newly_exposed_product_ids:string[]; overlap_product_ids:string[]; product_ids:string[]; supplier_ids:string[]; warehouse_ids:string[]; location_ids:string[]; alternative_warehouse_candidates:Array<Record<string,unknown>>; alternative_supplier_candidates:Array<Record<string,unknown>>; assumptions:string[]; provenance:string; inventory_relationship_available:boolean; catalog_only:boolean; catalog_association_count:number; affected_product_ids:string[]; affected_product_count:number; affected_supplier_ids:string[]; affected_supplier_count:number; current_risk_score:number|null; current_risk_level:string; impact_status:string; limitations:string[]; evidence_basis:string[]; explanation:string }
export function simulateScenario(input:{scenario_type:string;target_node_id:string;duration_hours:number;scenario_severity:string}){return request<ScenarioResponse>('/scenario/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(input)})}
export function explainScenarioCandidate(input:{candidate_type:'warehouse'|'supplier';evidence:Record<string,unknown>}){return request<{answer:string;provenance:string;limitations:string[];explanation_mode:'LLM_GROUNDED'|'DETERMINISTIC_FALLBACK'}>('/scenario/explain',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(input)})}
export type MasterStatus = {write_enabled:boolean;database_initialized:boolean;draft_change_count:number;published_version:number;published_at:string;last_modified?:string|null;overlay_available:boolean;validation_error_count:number;validation_warning_count:number;entity_counts?:Record<string,number>}
export function getMasterStatus(){return request<MasterStatus>('/master/status')}
export function getMasterEntities(type:string){return request<Entity[]>(`/master/${type}`)}
export function getMasterBrands(){return request<Entity[]>('/master/brands')}
export function getMasterRelationships(){return request<Record<string,unknown>[]>('/master/relationships')}
export function createMasterRelationship(payload:Record<string,unknown>){return request<Record<string,unknown>>('/master/relationships',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})}
export function deactivateMasterRelationship(id:string){return request<Record<string,unknown>>(`/master/relationships/${encodeURIComponent(id)}/deactivate`,{method:'POST'})}
export function getMasterChangeLog(){return request<Record<string,unknown>[]>('/master/change-log')}
export function createMasterEntity(type:string,payload:Record<string,unknown>){return request<Record<string,unknown>>(`/master/${type}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})}
export function updateMasterEntity(type:string,id:string,payload:Record<string,unknown>){return request<Record<string,unknown>>(`/master/${type}/${encodeURIComponent(id)}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})}
export function deactivateMasterEntity(type:string,id:string){return request<Record<string,unknown>>(`/master/${type}/${encodeURIComponent(id)}/deactivate`,{method:'POST'})}
export function getMasterValidation(){return request<{errors:Array<Record<string,string>>;warnings:Array<Record<string,string>>}>('/master/validate',{method:'POST'})}
export function getMasterPreview(){return request<Record<string,unknown>>('/master/preview')}
export function publishMaster(){return request<Record<string,unknown>>('/master/publish',{method:'POST'})}
export async function previewMasterImport(type:string,file:File){const form=new FormData();form.append('file',file);return request<Record<string,unknown>>(`/master/import/preview?import_type=${encodeURIComponent(type)}`,{method:'POST',body:form})}
export function confirmMasterImport(batchId:string){return request<Record<string,unknown>>(`/master/import/${encodeURIComponent(batchId)}/confirm`,{method:'POST'})}
export function getGraphEntity(nodeId:string){return request<GraphEntity>(`/current/graph/entity/${encodeURIComponent(nodeId)}`)}
export function getHistoricalMetrics(){return request<Record<string,unknown>>('/historical/model-metrics')}
export function refreshExternalSignals(){return request<Record<string,unknown>>('/current/refresh',{method:'POST'})}
export async function getCurrentSnapshot():Promise<Snapshot>{
  const [summary,liveSituation,events,products,warehouses,suppliers,metadata] = await Promise.all([
    getCurrentSummary(),getLiveSituation(),getCurrentEvents(),getProducts(),getWarehouses(),getSuppliers(),getCurrentMetadata()
  ])
  const normalize = (items:Entity[], node_type:string) => items.map(item => {
    return {...item, Node_ID:item.node_id, node_type, Node_Name:item.name, Risk_Level:item.risk_level, Risk_Score:item.risk_score, Connected_Events:item.connected_events}
  })
  const normalizedProducts=normalize(products,'Product'), normalizedWarehouses=normalize(warehouses,'Warehouse'), normalizedSuppliers=normalize(suppliers,'Supplier')
  return {summary,liveSituation,events,products:normalizedProducts,warehouses:normalizedWarehouses,suppliers:normalizedSuppliers,entities:[...normalizedProducts,...normalizedWarehouses,...normalizedSuppliers],metadata,provenance:'DERIVED / RULE_BASED'}
}
export function askSupplyBeacon(query:string){return request<any>('/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query})},LONG_REQUEST_TIMEOUT_MS)}
export function askSupplyBeaconAgentic(query:string){return request<AgenticResponse>('/query/agentic',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query})},LONG_REQUEST_TIMEOUT_MS)}
