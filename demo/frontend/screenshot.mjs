// Capture README screenshots: arena mid-answer + dashboard.
import { chromium } from 'playwright'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })

// Arena: upload, ask, wait for all three cards to finish, then shoot.
await page.goto('http://localhost:8000/')
await page.setInputFiles('input[type=file]', '/tmp/test_clevr.png')
await page.fill('textarea', 'How many items are there in the image?')
await page.click('text=Ask all models')
await page.waitForFunction(
  () => document.querySelectorAll('.streaming-dot').length === 0,
  { timeout: 120000 },
)
await page.waitForTimeout(500)
await page.screenshot({ path: '../assets/arena.png' })
console.log('arena.png captured')

// Dashboard
await page.click('nav >> text=dashboard')
await page.waitForSelector('text=Transfer matrix')
await page.waitForTimeout(500)
await page.screenshot({ path: '../assets/dashboard.png', fullPage: true })
console.log('dashboard.png captured')

await browser.close()
