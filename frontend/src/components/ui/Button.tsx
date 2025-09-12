import React from 'react'

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'default' | 'outline' | 'ghost'
}

export function Button({ className = '', variant = 'default', ...props }: Props) {
  const base = 'inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none h-9 px-4 py-2'
  const styles = {
    default: 'bg-black text-white hover:bg-black/90 focus-visible:ring-black',
    outline: 'border border-gray-300 bg-white hover:bg-gray-50',
    ghost: 'hover:bg-gray-100',
  } as const
  return <button className={`${base} ${styles[variant]} ${className}`} {...props} />
}

