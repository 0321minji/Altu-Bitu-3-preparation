#include<iostream>
#include<map>
#include<vector>
#include<string>
using namespace std;

//��Ϳ����� ���������� Ȱ���Ѵٸ� �� �Լ����� �Ű������� ���� �Ѱ��־� ���ʿ��� �޸� ���� �Ͼ ���� �ֱ⿡ ���������� �����ؿ�!

vector<int> store_used_alphabet;//���� ���ĺ��� ������ ���� -> 
vector<string> words(3);
vector<pair<bool,int>> alphabet_to_number(26, make_pair(false, -1));//<���ĺ� ��뿩��, �Ҵ�� ����>
vector<bool> used_number(10, false);//���� ��� ����

long long wordToNumber(string word) {//word�� ���ڷ� �ٲٴ� �Լ�
	long long num = 0;
	for (int i = 0; i < word.length(); i++) {
		num = num * 10 + alphabet_to_number[word[i] - 'A'].second;
	}
	return num;
}

bool calc() {//���� �����ϴ��� ���θ� �Ǵ��ϴ� �Լ�
	vector<long long> number(3);//word�� ���ڷ� �ٲ� ����� ������ ����

	for (int i = 0; i < 3; i++) {
		number[i] = wordToNumber(words[i]);
	}

	if (number[0] + number[1] == number[2]) {
		return true;
	}
	return false;
}

void checkUsedAlphabet(string word) {//���ĺ��� ���ڷ� �ٲ� ����� �����ϴ� ���Ϳ� ���� ���ĺ��� �����ϴ� ���� ����� �Լ�
	for (int i = 0; i < word.length(); i++) {

		if (!alphabet_to_number[word[i] - 'A'].first) {//���ĺ��� �������� ���ٸ�
			alphabet_to_number[word[i] - 'A'].first = true;
			store_used_alphabet.push_back(word[i] - 'A');
		}
		
	}
}

void backtracking(int level) {

	if (level == store_used_alphabet.size()) {//�ѹ��� bactracking���� ���ĺ� �ϳ��� ���ڰ� �Ҵ�ǹǷ� store_used_alphabet������ ������� bactracking�� level�� ������ ��� ���ĺ��� ���ڰ� �Ҵ�� ��
		if (calc()) {
			cout << "YES";
			exit(0);
		}

	}

	for (int i = 0; i < 10; i++) {//���ĺ��� ���ʷ� ���� �Ҵ�
		if (!used_number[i]) {

			used_number[i] = true;
			alphabet_to_number[store_used_alphabet[level]].second = i;

			backtracking(level + 1);

			used_number[i] = false;
			alphabet_to_number[store_used_alphabet[level]].second = -1;

		}
	}
}

/*
* 1. main���� �ܾ �Է¹޾� checkUsedAlphabet�Լ��� ���� ���ĺ� üũ
* 2. backtrackig�Լ��� ���ĺ��� ���ʷ� ���ڸ� �Ҵ��ϰ�
* 3. ��� ���ĺ��� ���ڰ� �Ҵ� �Ǿ��� ��
* 4. calc �Լ��� ����� ������ �����ϴ��� Ȯ�� -> ������ �����ϸ� YES ����ϰ� ����
*/
int main() {

	for (int i = 0; i < 3; i++) {
		cin >> words[i];
		checkUsedAlphabet(words[i]);//�Է¹��� �ܾ ���� ���ĺ� üũ
	}
	
	if (store_used_alphabet.size() > 10) {//���ĺ� ������ 10�� �ʰ��� 
		cout << "NO";
		return 0;
	}

	backtracking(0);

	cout << "NO";

	return 0;
}