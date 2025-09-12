import React from 'react'
import { FileUploader } from '@/components/FileUploader'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Label } from '@/components/ui/Label'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { apiNaiveBayes, apiPreview } from '@/lib/api'

export default function NaiveBayesPage() {
  const [file, setFile] = React.useState<File | null>(null)
  const [preview, setPreview] = React.useState<any | null>(null)
  const [target, setTarget] = React.useState<string>('')
  const [variant, setVariant] = React.useState<'gaussian' | 'multinomial'>('gaussian')
  const [alpha, setAlpha] = React.useState(1)
  const [loading, setLoading] = React.useState(false)
  const [result, setResult] = React.useState<any | null>(null)

  React.useEffect(() => {
    if (!file) { setPreview(null); setTarget(''); return }
    apiPreview(file).then((p) => {
      setPreview(p)
      // heuristic: last column as target if categorical
      const last = p.headers[p.headers.length - 1]
      setTarget(last)
    }).catch(console.error)
  }, [file])

  async function onRun() {
    if (!file || !target) return
    setLoading(true)
    try {
      const res = await apiNaiveBayes(file, { target, variant, alpha })
      setResult(res)
    } catch (e) {
      console.error(e)
      alert('Lỗi khi chạy Naive Bayes')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="text-lg font-semibold">Naive Bayes</div>
          <div className="text-sm text-gray-500">Phân lớp theo Bayes ngây thơ</div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
            <div className="md:col-span-2">
              <Label>Tệp dữ liệu</Label>
              <FileUploader file={file} onChange={setFile} />
            </div>
            <div>
              <Label className="mb-1 block">Thuộc tính mục tiêu</Label>
              <select className="h-9 w-full rounded-md border border-gray-300 px-3 text-sm" value={target} onChange={(e) => setTarget(e.target.value)}>
                <option value="" disabled>-- Chọn cột mục tiêu --</option>
                {preview?.headers?.map((h: string) => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3">
            <div>
              <Label className="mb-1 block">Biến thể</Label>
              <div className="flex gap-4 text-sm">
                <label className="flex items-center gap-2">
                  <input type="radio" name="variant" checked={variant==='gaussian'} onChange={() => setVariant('gaussian')} /> Gaussian
                </label>
                <label className="flex items-center gap-2">
                  <input type="radio" name="variant" checked={variant==='multinomial'} onChange={() => setVariant('multinomial')} /> Multinomial
                </label>
              </div>
            </div>
            {variant === 'multinomial' && (
              <div>
                <Label className="mb-1 block">Alpha (Laplace)</Label>
                <Input type="number" min={0} step={0.1} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
              </div>
            )}
          </div>

          <div className="mt-4">
            <Button onClick={onRun} disabled={!file || !target || loading}>{loading ? 'Đang chạy...' : 'Huấn luyện & Đánh giá'}</Button>
          </div>
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <div className="font-semibold">Kết quả</div>
          </CardHeader>
          <CardContent>
            <div className="text-sm">Độ chính xác (train): <b>{(result.accuracy_train * 100).toFixed(2)}%</b></div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

