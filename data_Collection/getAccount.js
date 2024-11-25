const puppeteer = require('puppeteer');
const fs = require('fs');

const { ACCOUNTS_LITS_PATH } = require('./global');



async function scrapeTopAccounts() {
    // Launch Puppeteer with headless mode off for debugging
    const browser = await puppeteer.launch({
        headless: false, // Set to false so you can visually check if login or CAPTCHA is required
        executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    });
    
    const page = await browser.newPage();
    
    // Set a custom User-Agent to avoid bot detection
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36');

    // Navigate to the page that lists accounts 
    await page.goto('https://etherscan.io/accounts/1?ps=100', { waitUntil: 'networkidle2', timeout: 60000 });
    
    let allAddresses = []; // Array to store all addresses
    
    let hasNextPage = true; // Flag to control pagination
    
    while (hasNextPage) {
        // Wait for the table to load
        await page.waitForSelector('table.table');
        
        // Scrape data from the current page
        const addresses = await page.evaluate(() => {
            const rows = Array.from(document.querySelectorAll('table tbody tr'));
            return rows.map(row => {
                const addressElement = row.querySelector('td:nth-child(2) span[data-highlight-target]'); 
                const tagElement = row.querySelector('td:nth-child(3)'); // tag Name column
                const balanceElement = row.querySelector('td:nth-child(4)'); // Balance column
                const percentageElement = row.querySelector('td:nth-child(5)'); // Percentage column
                const txnCountElement = row.querySelector('td:nth-child(6)'); // Txn Count column
                
                return {
                    address: addressElement ? addressElement.getAttribute('data-highlight-target') : null,
                    tagName: tagElement ? tagElement.textContent.trim() : null,
                    balance: balanceElement ? balanceElement.textContent.trim() : null,
                    percentage: percentageElement ? percentageElement.textContent.trim() : null,
                    txnCount: txnCountElement ? txnCountElement.textContent.trim() : null
                };
            }).filter(row => row.address !== null); // Filter out null rows
        });
        
        allAddresses = allAddresses.concat(addresses); // Append the addresses to the array
        
        console.log(`Scraped ${addresses.length} addresses from this page`);

        // Check if there's a "Next" button to go to the next page
        const nextButton = false;

        // TODO: To simplify the validation process, the pagination functionality has been removed.
        // const nextButton = await page.$('a[aria-label="Next"]');
        if (nextButton) {
            // Click the next button
            await Promise.all([
                page.waitForNavigation({ waitUntil: 'networkidle2' }), // Wait for page navigation
                nextButton.click()
            ]);
        } else {
            hasNextPage = false; // No more pages
        }
    }
    
    console.log(`Scraped a total of ${allAddresses.length} addresses.`);

    // Save the data to a JSON file 
    fs.writeFileSync(ACCOUNTS_LITS_PATH, JSON.stringify(allAddresses, null, 2), 'utf-8');

    await browser.close();
}

// Start the scrape function
scrapeTopAccounts();
