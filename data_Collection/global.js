const path = require('path');

// 定义常量
const ACCOUNTS_LITS_PATH = path.resolve(__dirname, './data/etherscan_accounts.json'); ; // 文件路径

const CHECH_RESULT_PTAH = path.resolve(__dirname, './data/phishing_check_results_final-'); ; // 文件路径;

// 导出常量
module.exports = {
  ACCOUNTS_LITS_PATH,
  CHECH_RESULT_PTAH,
};