import React from 'react'
import { FileUploader } from '@/components/FileUploader'
import { Button } from '@/components/ui/Button'
import { Label } from '@/components/ui/Label'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { apiDecisionTree, apiPreview } from '@/lib/api'
import { SampleDatasets } from '@/components/SampleDatasets'
import { PreviewTable } from '@/components/PreviewTable'


export default function DecisionTreePage() {
  const [file, setFile] = React.useState<File | null>(null)
  const [preview, setPreview] = React.useState<any | null>(null)
  const [target, setTarget] = React.useState<string>('')
  const [result, setResult] = React.useState<any | null>(null)
  const [loading, setLoading] = React.useState(false)
  const [predictionValues, setPredictionValues] = React.useState<Record<string, string>>({})
  const [predictionResult, setPredictionResult] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (!file) {
      setPreview(null);
      setTarget('');
      setPredictionValues({});
      setPredictionResult(null);
      return;
    }
    apiPreview(file).then((p) => {
      setPreview(p)
      // Set target as the last column (default behavior)
      const last = p.headers[p.headers.length - 1]
      setTarget(last)

      // Initialize prediction values for feature columns only (exclude target)
      const initialValues: Record<string, string> = {}
      p.headers
        .filter((header: string) => header !== last)
        .forEach((header: string) => {
          initialValues[header] = ''
        })
      setPredictionValues(initialValues)
    }).catch(console.error)
  }, [file])


  async function makePrediction() {
    if (!file || !target) return

    // Get feature columns (exclude target column)
    const featureColumns = Object.keys(predictionValues).filter(col => col !== target)
    const hasEmptyValues = featureColumns.some(feature => !predictionValues[feature])

    if (hasEmptyValues) {
      alert('Vui lòng chọn giá trị cho tất cả các thuộc tính dự đoán')
      return
    }

    setLoading(true)
    try {
      // Build tree if not already built
      let treeResult = result
      if (!treeResult) {
        treeResult = await apiDecisionTree(file, { target })
        setResult(treeResult)
      }

      // Prepare feature values for prediction (exclude target column)
      const featureValues: Record<string, string> = {}
      featureColumns.forEach(col => {
        featureValues[col] = predictionValues[col]
      })

      // Simple tree traversal for prediction
      function predictWithTree(tree: any, values: Record<string, string>): string {
        if (!tree || typeof tree === 'string') {
          return tree || 'Unknown'
        }

        if (tree.type === 'leaf') {
          return tree.prediction || 'Unknown'
        }

        if (tree.type === 'split' && tree.feature && tree.children) {
          const featureValue = values[tree.feature]
          if (featureValue && tree.children[featureValue]) {
            return predictWithTree(tree.children[featureValue], values)
          }
        }

        return 'Unknown'
      }

      const prediction = predictWithTree(treeResult.tree, featureValues)
      setPredictionResult(prediction)
    } catch (e) {
      console.error(e)
      alert('Lỗi khi thực hiện dự đoán')
    } finally {
      setLoading(false)
    }
  }

  function getUniqueValues(columnName: string): string[] {
    if (!preview || !preview.sample) return []

    const values: string[] = preview.sample
      .map((row: any) => String(row[columnName] || ''))
      .filter((value: string) => value !== null && value !== undefined && value !== '')

    return Array.from(new Set(values)).sort()
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="text-lg font-semibold">Cây quyết định</div>
          <div className="text-sm text-gray-500">Phân lớp với ID3 (Entropy/Information Gain)</div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-4">
            <div>
              <Label>Tệp dữ liệu</Label>
              <FileUploader file={file} onChange={setFile} />
              <SampleDatasets
                onPick={(f) => setFile(f)}
                samples={[
                  { label: 'golf_decision_tree.csv (ID3 Golf Dataset)', path: '/data/golf_decision_tree.csv' },
                  { label: 'play_tennis.csv', path: '/data/play_tennis.csv' },
                  { label: 'decision_tree_loan.csv', path: '/data/decision_tree_loan.csv' },
                  { label: 'decision_tree_mushroom.csv', path: '/data/decision_tree_mushroom.csv' },
                ]}
              />
            </div>

            {preview && (
              <div>
                <Label className="mb-3 block font-semibold text-base">Thực hiện dự đoán:</Label>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
                  {preview.headers
                    .filter((columnName: string) => columnName !== target)
                    .map((columnName: string) => (
                      <div key={columnName}>
                        <Label className="mb-1 block font-medium">{columnName}</Label>
                        <select
                          className="w-full h-9 rounded-md border border-gray-300 px-3 text-sm"
                          value={predictionValues[columnName] || ''}
                          onChange={(e) => setPredictionValues(prev => ({
                            ...prev,
                            [columnName]: e.target.value
                          }))}
                        >
                          <option value="" disabled>-- Chọn {columnName} --</option>
                          {getUniqueValues(columnName).map((value) => (
                            <option key={value} value={value}>{value}</option>
                          ))}
                        </select>
                      </div>
                    ))}
                </div>

                <div className="space-y-4">
                  <div className="flex items-center gap-4">
                    <Button onClick={makePrediction} disabled={!file || !target || loading}>
                      {loading ? 'Đang dự đoán...' : 'Dự đoán'}
                    </Button>

                    {predictionResult && (
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">Kết quả dự đoán {target}:</span>
                        <div className="px-3 py-1 bg-green-100 text-green-800 rounded-md font-semibold">
                          {predictionResult}
                        </div>
                      </div>
                    )}
                  </div>

                  {predictionResult && (
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <div className="text-sm">
                        <strong>Chi tiết dự đoán:</strong>
                        <div className="mt-2 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                          {Object.entries(predictionValues)
                            .filter(([feature]) => feature !== target)
                            .map(([feature, value]) => (
                              <span key={feature} className="inline-block">
                                <span className="font-medium">{feature}:</span> {value}
                              </span>
                            ))}
                          <span className="inline-block col-span-full mt-2">
                            <span className="font-medium text-green-700">→ {target}:</span>
                            <span className="text-green-700 font-bold ml-1">{predictionResult}</span>
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

        </CardContent>
      </Card>

      {preview && (
        <Card>
          <CardHeader>
            <div className="font-semibold">Dữ liệu hiện tại</div>
          </CardHeader>
          <CardContent>
            <PreviewTable headers={preview.headers} rows={preview.sample} />
          </CardContent>
        </Card>
      )}

    </div>
  )
}
