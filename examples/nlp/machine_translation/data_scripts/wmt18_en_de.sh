wmt_dir=$1;
out_dir=$2;
bicleaner_model_path=$3;
bifixer_path=$4;

mkdir -p ${out_dir}
mkdir -p ${wmt_dir}
mkdir -p ${wmt_dir}/orig

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz"
    "http://data.statmt.org/wmt18/translation-task/rapid2016.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
)

FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v13.tgz"
    "rapid2016.tgz"
    "dev.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training-parallel-nc-v13/news-commentary-v13.de-en"
    "rapid2016.de-en"
)

URLS_mono_de=(
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.de.shuffled.gz"
    "http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.de.shuffled.v2.gz"
    "http://data.statmt.org/wmt16/translation-task/news.2015.de.shuffled.gz"
    "http://data.statmt.org/wmt17/translation-task/news.2016.de.shuffled.gz"
    "http://data.statmt.org/wmt18/translation-task/news.2017.de.shuffled.deduped.gz"
)

URLS_mono_en=(
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.en.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.en.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.en.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.en.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.en.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz"
    "http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.en.shuffled.v2.gz"
    "http://data.statmt.org/wmt16/translation-task/news.2015.en.shuffled.gz"
    "http://data.statmt.org/wmt17/translation-task/news.2016.en.shuffled.gz"
    "http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.deduped.gz"
)

FILES_de=(
    "news.2007.de.shuffled.gz"
    "news.2008.de.shuffled.gz"
    "news.2009.de.shuffled.gz"
    "news.2010.de.shuffled.gz"
    "news.2011.de.shuffled.gz"
    "news.2012.de.shuffled.gz"
    "news.2013.de.shuffled.gz"
    "news.2014.de.shuffled.v2.gz"
    "news.2015.de.shuffled.gz"
    "news.2016.de.shuffled.gz"
    "news.2017.de.shuffled.deduped.gz"
)

FILES_en=(
    "news.2007.en.shuffled.gz"
    "news.2008.en.shuffled.gz"
    "news.2009.en.shuffled.gz"
    "news.2010.en.shuffled.gz"
    "news.2011.en.shuffled.gz"
    "news.2012.en.shuffled.gz"
    "news.2013.en.shuffled.gz"
    "news.2014.en.shuffled.v2.gz"
    "news.2015.en.shuffled.gz"
    "news.2016.en.shuffled.gz"
    "news.2017.en.shuffled.deduped.gz"
)

OUTDIR=$out_dir
lang1=en
lang2=de
lang=en-de
rev_lang=de-en
orig=${wmt_dir}/orig

mkdir -p $OUTDIR
mkdir -p $OUTDIR/parallel
mkdir -p $OUTDIR/mono

cd $orig

echo "=================================================="
echo "========= Downloading and Unpacking Data ========="
echo "=================================================="

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit 1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done

echo "pre-processing train data..."
rm $OUTDIR/parallel/*
for l in $lang1 $lang2; do
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l >> $OUTDIR/parallel/train.$lang.$l
    done
done

echo "Fetching Validation data $lang" 
sacrebleu -t wmt13 -l $lang --echo src > ${OUTDIR}/parallel/newstest2013-$lang.src
sacrebleu -t wmt13 -l $lang --echo ref > ${OUTDIR}/parallel/newstest2013-$lang.ref

echo "Fetching Test data $lang" 
sacrebleu -t wmt14 -l $lang --echo src > ${OUTDIR}/parallel/newstest2014-$lang.src
sacrebleu -t wmt14 -l $lang --echo ref > ${OUTDIR}/parallel/newstest2014-$lang.ref

echo "Fetching Validation data $rev_lang" 
sacrebleu -t wmt13 -l $rev_lang --echo src > ${OUTDIR}/parallel/newstest2013-$rev_lang.src
sacrebleu -t wmt13 -l $rev_lang --echo ref > ${OUTDIR}/parallel/newstest2013-$rev_lang.ref

echo "Fetching Test data $rev_lang" 
sacrebleu -t wmt14 -l $rev_lang --echo src > ${OUTDIR}/parallel/newstest2014-$rev_lang.src
sacrebleu -t wmt14 -l $rev_lang --echo ref > ${OUTDIR}/parallel/newstest2014-$rev_lang.ref

echo "=================================================="
echo "========= Filtering and Cleaning Data ============"
echo "=================================================="

if [ ! -f clean-corpus-n.perl ]
then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/training/clean-corpus-n.perl
    chmod +x clean-corpus-n.perl
fi

if [ ! -f normalize-punctuation.perl ]
then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/normalize-punctuation.perl
    chmod +x normalize-punctuation.perl
fi

if [ ! -f remove-non-printing-char.perl ]
then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/remove-non-printing-char.perl
    chmod +x remove-non-printing-char.perl
fi

echo "Filtering data based on max length and length ratio ..."
./clean-corpus-n.perl -ratio 1.3 ${OUTDIR}/parallel/train.$lang $lang1 $lang2 ${OUTDIR}/parallel/train.$lang.filter 1 250

echo "Applying bi-cleaner classifier"
awk '{print "-\t-"}' $OUTDIR/parallel/train.$lang.filter.en \
| paste -d "\t" - $OUTDIR/parallel/train.$lang.filter.en $OUTDIR/parallel/train.$lang.filter.de \
| bicleaner-classify - - $bicleaner_model_path > $OUTDIR/parallel/train.$lang.bicleaner.score

echo "Applying bifixer & dedup"
cat $OUTDIR/parallel/train.$lang.bicleaner.score \
| parallel -j 19 --pipe -k -l 30000 python $bifixer_path/bifixer.py \
    --ignore_segmentation -q - - en de \
    | awk -F "\t" '!seen[$6]++' - > $OUTDIR/parallel/train.$lang.bifixer.score

awk -F "\t" '{ if ($5>0.5) {print $3}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.50.en
awk -F "\t" '{ if ($5>0.5) {print $4}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.50.de

echo "Normalizing punct ..."
for l in $lang1 $lang2; do
    cat $OUTDIR/parallel/train.$lang.50.$l | perl normalize-punctuation.perl -l $l | perl remove-non-printing-char.perl > $OUTDIR/parallel/train.clean.$lang.50.$l
done

echo "Creating shared data for vocab creation ..."
cat $OUTDIR/parallel/train.clean.$lang.50.$lang1 $OUTDIR/parallel/train.clean.$lang.50.$lang2 > $OUTDIR/parallel/train.clean.$lang.50.common

echo "Normalizing valid/test punct ..."
cat ${OUTDIR}/parallel/newstest2013-$lang.src | perl normalize-punctuation.perl -l en | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2013-$lang.clean.src
cat ${OUTDIR}/parallel/newstest2013-$lang.ref | perl normalize-punctuation.perl -l de | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2013-$lang.clean.ref

cat ${OUTDIR}/parallel/newstest2014-$lang.src | perl normalize-punctuation.perl -l en | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2014-$lang.clean.src
cat ${OUTDIR}/parallel/newstest2014-$lang.ref | perl normalize-punctuation.perl -l de | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2014-$lang.clean.ref

cat ${OUTDIR}/parallel/newstest2013-$rev_lang.src | perl normalize-punctuation.perl -l de | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2013-$rev_lang.clean.src
cat ${OUTDIR}/parallel/newstest2013-$rev_lang.ref | perl normalize-punctuation.perl -l en | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2013-$rev_lang.clean.ref

cat ${OUTDIR}/parallel/newstest2014-$rev_lang.src | perl normalize-punctuation.perl -l de | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2014-$rev_lang.clean.src
cat ${OUTDIR}/parallel/newstest2014-$rev_lang.ref | perl normalize-punctuation.perl -l en | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2014-$rev_lang.clean.ref

echo 'Shuffling parallel data ...'
shuf --random-source=$OUTDIR/parallel/train.clean.$lang.50.$lang1 $OUTDIR/parallel/train.clean.$lang.50.$lang1 > $OUTDIR/parallel/train.clean.$lang.50.$lang1.shuffled
shuf --random-source=$OUTDIR/parallel/train.clean.$lang.50.$lang1 $OUTDIR/parallel/train.clean.$lang.50.$lang2 > $OUTDIR/parallel/train.clean.$lang.50.$lang2.shuffled

echo "=================================================="
echo "========== Fetching Monolingual Data ============="
echo "=================================================="

OUTDIR_MONO=$OUTDIR/mono/
mkdir -p $OUTDIR_MONO

cd $orig

echo "Done Processing Parallel Corpus, Fetching Monolingual Data ..."

echo "Fetching English Monolingual data ..."

for ((i=0;i<${#URLS_mono_en[@]};++i)); do
    file=${FILES_en[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS_mono_en[i]}
        wget "$url"
    fi
done

if [ -f ${OUTDIR_MONO}/monolingual.news.en ]; then
    echo "found monolingual sample, skipping shuffle/sample/tokenize"
else
    gzip -c -d -k $(for FILE in "${FILES_en[@]}"; do echo $orig/$FILE; done) > $OUTDIR_MONO/monolingual.news.en
fi

echo "Deduplicating data ..."
if [ -f ${OUTDIR_MONO}/monolingual.news.dedup.en ]; then
    echo "found deduplicated monolingual sample, skipping deduplication step"
else
    awk '!a[$0]++' ${OUTDIR_MONO}/monolingual.news.en > ${OUTDIR_MONO}/monolingual.news.dedup.en
    echo "Cleaning data ..."
    cat ${OUTDIR_MONO}/monolingual.news.dedup.en | perl normalize-punctuation.perl -l en | perl remove-non-printing-char.perl > ${OUTDIR_MONO}/monolingual.news.dedup.clean.en
fi

echo "Fetching German Monolingual data ..."

cd $orig

for ((i=0;i<${#URLS_mono_de[@]};++i)); do
    file=${FILES_de[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS_mono_de[i]}
        wget "$url"
    fi
done

echo "Subsampling data ..."
if [ -f ${OUTDIR_MONO}/monolingual.news.de ]; then
    echo "found monolingual sample, skipping shuffle/sample/tokenize"
else
    gzip -c -d -k $(for FILE in "${FILES_de[@]}"; do echo $orig/$FILE; done) > ${OUTDIR_MONO}/monolingual.news.de
fi

echo "Deduplicating data ..."
if [ -f ${OUTDIR_MONO}/monolingual.news.dedup.de ]; then
    echo "found deduplicated monolingual sample, skipping deduplication step"
else
    awk '!a[$0]++' ${OUTDIR_MONO}/monolingual.news.de > ${OUTDIR_MONO}/monolingual.news.dedup.de
    echo "Cleaning data ..."
    cat ${OUTDIR_MONO}/monolingual.news.dedup.de | perl normalize-punctuation.perl -l de | perl remove-non-printing-char.perl > ${OUTDIR_MONO}/monolingual.news.dedup.clean.de
fi
