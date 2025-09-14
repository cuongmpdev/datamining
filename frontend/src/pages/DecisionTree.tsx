import React from 'react'
import { FileUploader } from '@/components/FileUploader'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Label } from '@/components/ui/Label'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { apiDecisionTree, apiPreview } from '@/lib/api'
import { SampleDatasets } from '@/components/SampleDatasets'

function TreeView({ node, depth = 0 }: { node: any; depth?: number }) {
  const pad = Array(depth).fill('│  ').join('')
  if (!node) return null
  if (node.type === 'leaf') {
    return <pre className="text-xs">{pad}└─ (leaf) → {String(node.prediction)}</pre>
  }
  if (node.is_numeric) {
    return (
      <div className="text-xs">
        <pre>{pad}└─ [{node.feature} ≤ {node.threshold?.toFixed?.(4) ?? node.threshold}]</pre>
        <TreeView node={node.left} depth={depth + 1} />
        <pre>{pad}└─ [{node.feature}  {node.threshold?.toFixed?.(4) ?? node.threshold}]</pre>
        <TreeView node={node.right} depth={depth + 1} />
      </div>
    )
  }
  return (
    <div className="text-xs">
      <pre>{pad}└─ split on {node.feature}</pre>
      {Object.entries(node.children || {}).map(([val, child]: any) => (
        <div key={String(val)}>
          <pre>{pad}  ├─ {String(val)}</pre>
          <TreeView node={child} depth={depth + 1} />
        </div>
      ))}
    </div>
  )
}

export default function DecisionTreePage() {
  const [file, setFile] = React.useState<File | null>(null)
  const [preview, setPreview] = React.useState<any | null>(null)
  const [target, setTarget] = React.useState<string>('')
  const [maxDepth, setMaxDepth] = React.useState<number | ''>('')
  const [minSplit, setMinSplit] = React.useState(2)
  const [result, setResult] = React.useState<any | null>(null)
  const [loading, setLoading] = React.useState(false)

  React.useEffect(() => {
    if (!file) { setPreview(null); setTarget(''); return }
    apiPreview(file).then((p) => {
      setPreview(p)
      const last = p.headers[p.headers.length - 1]
      setTarget(last)
    }).catch(console.error)
  }, [file])

  async function onRun() {
    if (!file || !target) return
    setLoading(true)
    try {
      const res = await apiDecisionTree(file, { target, max_depth: maxDepth === '' ? null : maxDepth, min_samples_split: minSplit })
      setResult(res)
    } catch (e) {
      console.error(e)
      alert('Lỗi khi chạy Cây quyết định')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="text-lg font-semibold">Cây quyết định</div>
          <div className="text-sm text-gray-500">Phân lớp với ID3 (Entropy/Information Gain)</div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
            <div className="md:col-span-2">
              <Label>Tệp dữ liệu</Label>
              <FileUploader file={file} onChange={setFile} />
              <SampleDatasets
                onPick={(f) => setFile(f)}
                samples={[
                  { label: 'play_tennis.csv', path: '/data/play_tennis.csv' },
                  { label: 'decision_tree_loan.csv', path: '/data/decision_tree_loan.csv' },
                  { label: 'decision_tree_mushroom.csv', path: '/data/decision_tree_mushroom.csv' },
                ]}
              />
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
              <Label className="mb-1 block">Độ sâu tối đa</Label>
              <Input type="number" min={1} value={maxDepth} onChange={(e) => setMaxDepth(e.target.value === '' ? '' : Number(e.target.value))} />
            </div>
            <div>
              <Label className="mb-1 block">Min samples split</Label>
              <Input type="number" min={2} value={minSplit} onChange={(e) => setMinSplit(Number(e.target.value))} />
            </div>
          </div>

          <div className="mt-4">
            <Button onClick={onRun} disabled={!file || !target || loading}>{loading ? 'Đang chạy...' : 'Huấn luyện & Xây cây'}</Button>
          </div>
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <div className="font-semibold">Kết quả</div>
          </CardHeader>
          <CardContent>
            <div className="text-sm mb-2">Độ chính xác (train): <b>{(result.accuracy_train * 100).toFixed(2)}%</b></div>
            <TreeView node={result.tree} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}
