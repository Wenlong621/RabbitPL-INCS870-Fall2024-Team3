const fetch = require('node-fetch');
const fs = require('fs'); 
const { Parser } = require('json2csv');
const path = require('path');

const { ACCOUNTS_LITS_PATH } = require('./global');

// Register an account on https://etherscan.io/ and obtain an API key.
const apiKey = 'YOUR API KEY';

const getList = async (address) => {
  try {
    // etherscan api
    const url = `https://api.etherscan.io/api?module=account&action=txlist&address=${address}&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=${apiKey}`;
    
    const response = await fetch(url);
   
    const data = await response.json();
  
    // Check the data status and save it to a JSON file
    if (data.status === "1" && data.message === "OK") {
        
        const resMap = {
            'blockHash': 'TxHash',
            'blockNumber': 'BlockHeight',
            'timeStamp': 'TimeStamp',
            'from': 'From',
            'to': 'To',
            'value':'Value',
            'contractAddress': 'ContractAddress',
            'input': 'Input',
            'isError': 'isError'
        }
        const fields = Object.keys(resMap) || [];
        const values = Object.values(resMap);
        const dataList = data.result?.map(itm => {
            const data = {};
            fields.forEach(attr => {
                data[resMap[attr]] = itm[attr]
            });
            return data;
        }) || [];
        const json2csvParser = new Parser({ values });
        // Convert to CSV.
        const csv = json2csvParser.parse(dataList);

        // Save as a CSV file.
        fs.writeFileSync(path.resolve(__dirname, `./data/${address}.csv`), csv);
        console.log(`Data saved to ${address}.csv`);
    } else {
    console.error("Error:", data.message);
    }
  } catch (error) {
    console.error("Fetch error:", error);
  }
}

// Verify addresses. be achieved by reading from a file.
const launch = () => {
    const data = fs.readFileSync(ACCOUNTS_LITS_PATH, 'utf8');
    const accounts = JSON.parse(data);
    const suspiciousAccounts = accounts.filter(itm => itm.tagName.includes('Fake_Phishing'))

    // To test the process, the first data entry is selected by default. However, specific addresses can be queried as needed.
    const address = (suspiciousAccounts[0] || accounts[0])?.address;

    if(address) {
        getList(address)
    }
}

launch();