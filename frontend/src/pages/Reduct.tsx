import React from 'react'
import { FileUploader } from '@/components/FileUploader'
import { Button } from '@/components/ui/Button'
import { Label } from '@/components/ui/Label'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { apiPreview, apiReduct } from '@/lib/api'

export default function ReductPage() {
  const [file, setFile] = React.useState<File | null>(null)
  const [preview, setPreview] = React.useState<any | null>(null)
  const [decision, setDecision] = React.useState<string>('')
  const [conditional, setConditional] = React.useState<string[]>([])
  const [loading, setLoading] = React.useState(false)
  const [result, setResult] = React.useState<any | null>(null)

  React.useEffect(() => {
    if (!file) { setPreview(null); setDecision(''); setConditional([]); return }
    apiPreview(file).then((p) => {
      setPreview(p)
      const last = p.headers[p.headers.length - 1]
      setDecision(last)
      setConditional(p.headers.filter((h: string) => h !== last))
    }).catch(console.error)
  }, [file])

  async function onRun() {
    if (!file || !decision) return
    setLoading(true)
    try {
      const res = await apiReduct(file, { decision, conditional: conditional.join(',') })
      setResult(res)
    } catch (e) {
      console.error(e)
      alert('Lỗi khi tính Reduct')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="text-lg font-semibold">Rough Set Reduct</div>
          <div className="text-sm text-gray-500">Giảm thuộc tính bằng QuickReduct</div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
            <div className="md:col-span-2">
              <Label>Tệp dữ liệu</Label>
              <FileUploader file={file} onChange={setFile} />
            </div>
            <div>
              <Label className="mb-1 block">Thuộc tính quyết định</Label>
              <select className="h-9 w-full rounded-md border border-gray-300 px-3 text-sm" value={decision} onChange={(e) => setDecision(e.target.value)}>
                <option value="" disabled>-- Chọn cột --</option>
                {preview?.headers?.map((h: string) => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
            </div>
          </div>

          {preview && (
            <div className="mt-3">
              <Label className="mb-2 block">Chọn thuộc tính điều kiện</Label>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                {preview.headers.filter((h: string) => h !== decision).map((h: string) => (
                  <label key={h} className="flex items-center gap-2">
                    <input type="checkbox" checked={conditional.includes(h)} onChange={(e) => setConditional((prev) => e.target.checked ? [...prev, h] : prev.filter((x) => x !== h))} />
                    {h}
                  </label>
                ))}
              </div>
            </div>
          )}

          <div className="mt-4">
            <Button onClick={onRun} disabled={!file || !decision || loading}>{loading ? 'Đang chạy...' : 'Tính Reduct'}</Button>
          </div>
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <div className="font-semibold">Kết quả</div>
          </CardHeader>
          <CardContent>
            <div className="text-sm">Reduct: <b>{result.reduct.join(', ')}</b></div>
            <div className="text-sm">γ(R): <b>{result.gamma_R.toFixed(4)}</b> | γ(C): <b>{result.gamma_C.toFixed(4)}</b></div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

