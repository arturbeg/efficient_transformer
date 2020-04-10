cd data
echo "- Downloading 1B words"

mkdir -p one-billion-words
cd one-billion-words

wget --no-proxy http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar xzvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz

path="1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/"
cat ${path}/news.en.heldout-00000-of-00050 > valid.txt
cat ${path}/news.en.heldout-00000-of-00050 > test.txt

wget https://github.com/rafaljozefowicz/lm/raw/master/1b_word_vocab.txt

cd ..
echo "---"
echo "Happy language modeling :)"