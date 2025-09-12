import React from 'react'
import { Button } from './ui/Button'

export function FileUploader({ file, onChange }: { file: File | null; onChange: (file: File | null) => void }) {
  const inputRef = React.useRef<HTMLInputElement>(null)
  return (
    <div className="flex items-center gap-2">
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        className="hidden"
        onChange={(e) => onChange(e.target.files?.[0] || null)}
      />
      <Button variant="outline" onClick={() => inputRef.current?.click()}>Chọn CSV</Button>
      <div className="text-sm text-gray-600">{file ? file.name : 'Chưa chọn tệp'}</div>
      {file && (
        <Button variant="ghost" onClick={() => onChange(null)}>Xóa</Button>
      )}
    </div>
  )
}

