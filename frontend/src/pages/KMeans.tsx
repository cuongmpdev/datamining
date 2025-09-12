import React from 'react'
import { FileUploader } from '@/components/FileUploader'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Label } from '@/components/ui/Label'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { apiKMeans, apiPreview } from '@/lib/api'
import { PreviewTable } from '@/components/PreviewTable'

export default function KMeansPage() {
  const [file, setFile] = React.useState<File | null>(null)
  const [preview, setPreview] = React.useState<any | null>(null)
  const [k, setK] = React.useState(3)
  const [columns, setColumns] = React.useState<string[]>([])
  const [loading, setLoading] = React.useState(false)
  const [result, setResult] = React.useState<any | null>(null)

  React.useEffect(() => {
    if (!file) { setPreview(null); setColumns([]); return }
    apiPreview(file).then((p) => {
      setPreview(p)
      const numeric = Object.entries(p.types as Record<string, string>)
        .filter(([_, t]) => t === 'numeric')
        .map(([h]) => h)
      setColumns(numeric)
    }).catch((e) => console.error(e))
  }, [file])

  async function onRun() {
    if (!file || !preview) return
    setLoading(true)
    try {
      const res = await apiKMeans(file, { k, columns: columns.join(',') })
      setResult(res)
    } catch (e) {
      console.error(e)
      alert('Lỗi khi chạy KMeans')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="text-lg font-semibold">K-Means Clustering</div>
          <div className="text-sm text-gray-500">Chọn tệp CSV và tham số</div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
            <div className="md:col-span-2">
              <Label>Tệp dữ liệu</Label>
              <FileUploader file={file} onChange={setFile} />
            </div>
            <div>
              <Label className="mb-1 block">Số cụm (k)</Label>
              <Input type="number" min={1} value={k} onChange={(e) => setK(Number(e.target.value))} />
            </div>
          </div>

          {preview && (
            <div className="mt-4">
              <div className="font-medium mb-2">Chọn thuộc tính số dùng để gom cụm</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {preview.headers.map((h: string) => (
                  <label key={h} className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={columns.includes(h)}
                      disabled={preview.types[h] !== 'numeric'}
                      onChange={(e) => {
                        setColumns((prev) => e.target.checked ? [...prev, h] : prev.filter((x) => x !== h))
                      }}
                    />
                    <span>{h} {preview.types[h] !== 'numeric' && <em className="text-gray-400">(không phải số)</em>}</span>
                  </label>
                ))}
              </div>
            </div>
          )}

          <div className="mt-4">
            <Button onClick={onRun} disabled={!file || loading}>{loading ? 'Đang chạy...' : 'Chạy K-Means'}</Button>
          </div>
        </CardContent>
      </Card>

      {preview && (
        <Card>
          <CardHeader>
            <div className="font-semibold">Xem nhanh dữ liệu</div>
          </CardHeader>
          <CardContent>
            <PreviewTable headers={preview.headers} rows={preview.sample} />
          </CardContent>
        </Card>
      )}

      {result && (
        <Card>
          <CardHeader>
            <div className="font-semibold">Kết quả</div>
          </CardHeader>
          <CardContent>
            <div className="text-sm">Inertia: <b>{result.inertia.toFixed(4)}</b> | Lặp: {result.iterations}</div>
            <div className="mt-3 space-y-2">
              <div className="font-medium">Tâm cụm:</div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {result.centroids.map((c: number[], i: number) => (
                  <div key={i} className="rounded border p-2 text-sm bg-gray-50">
                    Cụm {i}: [{c.map((v) => v.toFixed(4)).join(', ')}]
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

