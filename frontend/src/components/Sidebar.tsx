import React from 'react'
import { Button } from './ui/Button'

type Item = {
  key: string
  label: string
}

export function Sidebar({ items, active, onSelect }: { items: Item[]; active: string; onSelect: (key: string) => void }) {
  return (
    <aside className="sidebar fixed left-0 top-0 bottom-0 border-r border-gray-200 bg-white">
      <div className="px-4 py-3 border-b">
        <div className="font-semibold">Data Mining Toolkit</div>
        <div className="text-xs text-gray-500">React + FastAPI</div>
      </div>
      <nav className="p-2 space-y-1">
        {items.map((it) => (
          <Button
            key={it.key}
            variant={active === it.key ? 'default' : 'ghost'}
            className="w-full justify-start"
            onClick={() => onSelect(it.key)}
          >
            {it.label}
          </Button>
        ))}
      </nav>
      <div className="absolute bottom-0 left-0 right-0 p-3 border-t text-xs text-gray-500">
        <div>Upload CSV, chọn tham số, chạy thuật toán.</div>
      </div>
    </aside>
  )
}

