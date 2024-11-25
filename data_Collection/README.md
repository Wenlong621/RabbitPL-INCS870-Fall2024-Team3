
# Data Collection

This project is a Node.js-based application designed to scrape Ethereum account data, including both legitimate and suspicious accounts, for further analysis. It includes three main functionalities:

1. **getAccount**: Extracts Ethereum account data from the blockchain explorer.
2. **checkPhishing**: Validates accounts to determine if they are suspicious.
3. **getTransactions**: Fetches transaction records associated with specific accounts.

---

## **Prerequisites**

- Install **Node.js** (version **22.9.0** or higher).
- Ensure you have a stable internet connection to access Ethereum account data.
- Create an account on **Etherscan** and obtain an **API key** for fetching transaction data.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd 870-PhishingScamDetectionOnEthereum
   ```

2. Install the dependencies:
   ```bash
   npm install
   ```

3. Set up your **Etherscan API key**:
   - Create an account on [Etherscan](https://etherscan.io/).
   - Obtain your API key and replace the value of the apiKey variable in the getTransactions.js file.

---

## **Usage**

### **1. Execute `getAccount` (Mandatory Step)**
Before running the other modules, you must first extract Ethereum account data by executing the `getAccount` script:

```bash
npm run account
```

This will generate a JSON file with account data in the `data` directory.

#### **Notes**:
- **Pagination Disabled**: For demonstration purposes, pagination has been disabled in the `getAccount.js` file. Specifically:
  ```javascript
  // const nextButton = await page.$('a[aria-label="Next"]');
  nextButton = false; // Disable pagination for quick results
  ```
  - To enable pagination for full data extraction, uncomment the `nextButton` line and remove `nextButton = false`.

---

### **2. Execute `checkPhishing`**
Once the account data is available, check for suspicious accounts:

```bash
npm run check
```

#### **Notes**:
- The execution time is limited to **5 seconds** for demonstration purposes, which may not process all accounts. To analyze all 10,000 accounts, adjust the code in `checkPhishing.js`


### **3. Execute `getTransactions`**
Finally, fetch transaction records for the accounts:

```bash
npm run trans
```

#### **Notes**:
- The script uses the **first account** (either suspicious or normal) for testing by default. To query a specific account, modify the logic in `getTransactions.js`:
  ```javascript
  const address = (suspiciousAccounts[0] || accounts[0])?.address;// Modify as needed
  ```
- **Etherscan API Requirement**: Ensure your **Etherscan API key** is properly setted

---

## **Scripts Overview**

- **`npm run account`**: Extracts Ethereum account data.
- **`npm run check`**: Validates accounts for phishing activity (depends on `getAccount` data).
- **`npm run trans`**: Fetches transaction records (depends on `getAccount` data).

> **Important**: Always run `npm run account` before executing `npm run check` or `npm run trans`.

---

## **Project Structure**

```plaintext
data_Collection/
├── data/                  # Directory for storing generated data
├── checkPhishing.js       # Script for detecting suspicious accounts
├── getAccount.js          # Script for extracting Ethereum account data
├── getTransactions.js     # Script for fetching transaction records
├── global.js              # Shared utilities and constants
└── .gitkeep               # Keeps the data directory in the repository
```


## **Contact**

If you encounter any issues, feel free to open an issue or reach out to the project maintainer.
