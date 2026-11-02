// Capture the README screenshots from a running server (port 8000): the examples and results views.
import { chromium } from 'playwright'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })

await page.goto('http://localhost:8000/')
await page.waitForSelector('article img')
await page.waitForTimeout(500)
await page.screenshot({ path: '../assets/gallery.png' })
console.log('gallery.png captured')

await page.getByRole('button', { name: 'Results' }).click()
await page.waitForSelector('text=Answer accuracy')
await page.waitForTimeout(500)
await page.screenshot({ path: '../assets/results.png' })
console.log('results.png captured')

await browser.close()
