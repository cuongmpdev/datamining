import React from 'react'
import KMeansPage from '@/pages/KMeans'
import NaiveBayesPage from '@/pages/NaiveBayes'
import DecisionTreePage from '@/pages/DecisionTree'
import ReductPage from '@/pages/Reduct'
import { Sidebar } from '@/components/Sidebar'

const items = [
  { key: 'kmeans', label: 'Clustering (K-Means)' },
  { key: 'bayes', label: 'Bayes (Naive Bayes)' },
  { key: 'tree', label: 'Cây quyết định' },
  { key: 'reduct', label: 'Reduct (Rough Set)' },
]

export default function App() {
  const [active, setActive] = React.useState(items[0].key)
  return (
    <div>
      <Sidebar items={items} active={active} onSelect={setActive} />
      <main className="content p-4 max-w-6xl mx-auto">
        {active === 'kmeans' && <KMeansPage />}
        {active === 'bayes' && <NaiveBayesPage />}
        {active === 'tree' && <DecisionTreePage />}
        {active === 'reduct' && <ReductPage />}
      </main>
    </div>
  )}

