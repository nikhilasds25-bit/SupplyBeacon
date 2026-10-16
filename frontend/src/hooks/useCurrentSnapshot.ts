import {useEffect, useState} from 'react'
import {getCurrentSnapshot, Snapshot} from '../services/supplyBeaconApi'

export function useCurrentSnapshot() {
  const [data, setData] = useState<Snapshot | null>(null)
  useEffect(() => {
    let mounted = true
    let pending = false
    const load = async (reset = false) => {
      if (pending) return
      pending = true
      if (reset) setData(null)
      try {
        const snapshot = await getCurrentSnapshot()
        if (mounted) setData(snapshot)
      } catch (error) {
        if (mounted) setData({error: (error as Error).message, retry: () => {void load(true)}} as Snapshot)
      } finally {
        pending = false
      }
    }
    void load()
    const refreshNow = () => {void load()}
    window.addEventListener('supplybeacon-refresh-current', refreshNow)
    // Refresh entity risks as well as summary/events so cards and charts agree.
    const timer = window.setInterval(() => {void load()}, 60000)
    return () => {
      mounted = false
      window.clearInterval(timer)
      window.removeEventListener('supplybeacon-refresh-current', refreshNow)
    }
  }, [])
  return data
}
