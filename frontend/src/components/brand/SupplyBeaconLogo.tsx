import { Link } from 'react-router-dom'

type SupplyBeaconLogoProps = {
  variant?: 'compact' | 'full'
  className?: string
  to?: string
}

export function SupplyBeaconLogo({
  variant = 'full',
  className = '',
  to = '/dashboard',
}: SupplyBeaconLogoProps) {
  const mark = (
    <span className="supplybeacon-mark" aria-hidden="true">
      <svg viewBox="0 0 64 64" role="img">
        <defs>
          <linearGradient id="supplybeacon-gradient" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#ff8ab7" />
            <stop offset="0.52" stopColor="#ff4f91" />
            <stop offset="1" stopColor="#a855f7" />
          </linearGradient>
        </defs>
        <path d="M11 19c7-9 20-12 31-8 5 2 8 5 11 9-7-4-14-5-20-3-7 2-10 7-9 12 1 5 6 8 12 8h9c4 0 6 2 6 5 0 5-6 9-14 9-11 0-20-4-26-11 7 4 14 5 20 3 4-1 6-3 6-5 0-2-2-3-5-3h-8C14 35 8 28 11 19Z" fill="none" stroke="url(#supplybeacon-gradient)" strokeWidth="5" strokeLinecap="round" />
        <path d="M36 20v25M31 45h10M33 20h6l-2-7h-2l-2 7ZM29 21h14" fill="none" stroke="#f8fafc" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
        <path className="beacon-beam" d="M36 10V5M36 5l-3 3M36 5l3 3M40 22l17-7M32 22l-17-7" stroke="#ff6a9e" strokeWidth="1.7" strokeLinecap="round" opacity=".85" />
      </svg>
    </span>
  )

  const content = variant === 'full' ? (
    <>
      {mark}
      <span className="supplybeacon-wordmark"><b>Supply</b><strong>Beacon</strong></span>
    </>
  ) : mark

  return (
    <Link className={`supplybeacon-logo ${variant} ${className}`} to={to} aria-label="SupplyBeacon">
      {content}
    </Link>
  )
}
