// 分析申請者的面試申請情況

import Papa from 'papaparse';
import fs from 'fs/promises';

const fileContent = await fs.readFile('scenario_7.csv', { encoding: 'utf8' });

const csvData = Papa.parse(fileContent, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true
});

console.log("=== 面試數量理論分析 ===");
console.log();

// 統計每位申請者申請了哪些部門
const applicantDeptMap = {};

csvData.data.forEach(row => {
    const applicantId = row.ID;
    const dept = row.dept;
    
    if (!applicantDeptMap[applicantId]) {
        applicantDeptMap[applicantId] = {
            name: row.Name,
            departments: new Set(),
            timeSlots: 0
        };
    }
    
    applicantDeptMap[applicantId].departments.add(dept);
    applicantDeptMap[applicantId].timeSlots++;
});

console.log("申請者部門申請統計:");
console.log("==================");

let totalTheoreticalInterviews = 0;
const departmentCounts = {};
const applicantInterviewCounts = {};

Object.entries(applicantDeptMap).forEach(([id, info]) => {
    const numDepts = info.departments.size;
    const deptList = Array.from(info.departments).sort();
    
    totalTheoreticalInterviews += numDepts;
    
    // 統計申請部門數量分佈
    if (!applicantInterviewCounts[numDepts]) {
        applicantInterviewCounts[numDepts] = 0;
    }
    applicantInterviewCounts[numDepts]++;
    
    // 統計各部門申請人數
    deptList.forEach(dept => {
        if (!departmentCounts[dept]) {
            departmentCounts[dept] = 0;
        }
        departmentCounts[dept]++;
    });
    
    console.log(`ID ${id} (${info.name}): 申請 ${numDepts} 個部門 [${deptList.join(', ')}]`);
});

console.log();
console.log("=== 統計摘要 ===");
console.log(`總申請者數量: ${Object.keys(applicantDeptMap).length}`);
console.log(`理論最大面試總數: ${totalTheoreticalInterviews}`);
console.log();

console.log("申請面試數量分佈:");
Object.entries(applicantInterviewCounts).sort((a,b) => Number(a[0]) - Number(b[0])).forEach(([numInterviews, count]) => {
    const percentage = (count / Object.keys(applicantDeptMap).length * 100).toFixed(1);
    console.log(`  ${numInterviews} 場面試: ${count} 人 (${percentage}%)`);
});

console.log();
console.log("各部門申請人數:");
Object.entries(departmentCounts).sort((a,b) => b[1] - a[1]).forEach(([dept, count]) => {
    console.log(`  ${dept}: ${count} 人申請`);
});

// console.log();
// console.log("=== 理論 vs 實際對比 ===");
// console.log(`理論最大面試數: ${totalTheoreticalInterviews}`);
// console.log(`當前實際安排: 94 場面試`);
// console.log(`達成率: ${(94 / totalTheoreticalInterviews * 100).toFixed(1)}%`);
// console.log(`缺少: ${totalTheoreticalInterviews - 94} 場面試`);

// console.log();
// console.log("=== 分析結論 ===");
// if (totalTheoreticalInterviews <= 152) {
//     console.log("✅ 理論上所有申請都可以被滿足（每人最多2場面試）");
// } else {
//     console.log("⚠️  理論面試數超過限制，需要優先順序決策");
// }

// // 檢查是否每個人都申請了所有部門
// const allDepts = ['AC', 'DM', 'GPDA', 'PR'];
// const applicantsWithAllDepts = Object.values(applicantDeptMap).filter(info => 
//     allDepts.every(dept => info.departments.has(dept))
// ).length;

// console.log();
// if (applicantsWithAllDepts === Object.keys(applicantDeptMap).length) {
//     console.log("📝 所有申請者都申請了全部4個部門");
//     console.log("   這意味著理論最大面試數 = 76人 × 2場/人 = 152場");
// } else {
//     console.log(`📝 有 ${applicantsWithAllDepts} 人申請了全部4個部門`);
//     console.log("   其他人申請了部分部門");
// }