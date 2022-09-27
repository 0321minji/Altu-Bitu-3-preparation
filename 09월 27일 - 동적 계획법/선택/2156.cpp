#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int maxWine(int n, vector<int> wine) {
	vector<int> dp(n + 1, 0);
	dp[1] = wine[1];
	dp[2] = wine[1] + wine[2];

	//max() �Լ� ��� �� 3�� �̻��� �� �񱳽� {}�� ������ �� ���� ��� �����մϴ�!

	for (int i = 3; i < n + 1; i++) {
		//���� ū ������ ����
		dp[i] = max({ dp[i - 3] + wine[i - 1] + wine[i],dp[i - 2] + wine[i],dp[i - 1] });
	}
	return dp[n];
}
int main() {
	int n;

	cin >> n;
	vector<int> wine(n + 1);


	for (int i = 1; i < n + 1; i++) {
		cin >> wine[i];
	}

	cout << maxWine(n, wine);
}