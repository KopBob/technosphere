import org.apache.commons.lang.StringEscapeUtils;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.hbase.Cell;
import org.apache.hadoop.hbase.CellUtil;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.Delete;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.*;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.hbase.util.Pair;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.yecht.Data;

import javax.annotation.Nonnull;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import java.net.URL;






public class HW3 extends Configured implements Tool {

    static enum Counters {
        ROBOTS_COUNTER
    }

    public static class TextPair implements WritableComparable<TextPair> {
        private Text first;
        private Text second;

        public TextPair() {
            set(new Text(), new Text());
        }

        public TextPair(String first, String second) {
            set(new Text(first), new Text(second));
        }

        public TextPair(Text first, Text second) {
            set(first, second);
        }

        private void set(Text a, Text b) {
            first = a;
            second = b;
        }

        public Text getFirst() {
            return first;
        }

        public Text getSecond() {
            return second;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            first.write(out);
            second.write(out);
        }

        @Override
        public int compareTo(@Nonnull TextPair o) {
            int cmp = first.compareTo(o.first);
            return (cmp == 0) ? second.compareTo(o.second) : cmp;
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            first.readFields(dataInput);
            second.readFields(dataInput);
        }

        @Override
        public int hashCode() {
            return first.hashCode() * 163 + second.hashCode();
        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof TextPair) {
                TextPair tp = (TextPair) obj;
                return first.equals(tp.first) && second.equals(tp.second);
            }
            return false;
        }

        @Override
        public String toString() {
            return first + "\t" + second;
        }
    }

    public int on_finish(String[] args) throws Exception {
        Job job = GetJobConf(args);
        System.out.println("NUMBER_OF_REDUCERS  num = " + job.getCounters().findCounter("NUMBER_OF_REDUCERS", "num").getValue());
        System.out.println("ROBOTS_COUNTER  num = " + job.getCounters().findCounter("ROBOTS_COUNTER", "num").getValue());
        System.out.println("ROBOTS_WITH_/  num = " + job.getCounters().findCounter("ROBOTS_WITH_/", "num").getValue());
        System.out.println("ROBOTS_WITH_*  num = " + job.getCounters().findCounter("ROBOTS_WITH_*", "num").getValue());
        System.out.println("ROBOTS_WITH_$  num = " + job.getCounters().findCounter("ROBOTS_WITH_$", "num").getValue());
        System.out.println("ROBOTS_WITH_ELSE  num = " + job.getCounters().findCounter("ROBOTS_WITH_ELSE", "num").getValue());

//        job.getCounters().findCounter("NUMBER_OF_REDUCERS", "num");
//        job.getCounters().findCounter("ROBOTS_COUNTER", "num");
//        job.getCounters().findCounter("ROBOTS_WITH_/", "num");
//        job.getCounters().findCounter("ROBOTS_WITH_*", "num");
//        job.getCounters().findCounter("ROBOTS_WITH_$", "num");
//        job.getCounters().findCounter("ROBOTS_WITH_ELSE", "num");
//        job.getCounters().findCounter("DISALLOWED_WITH_/", "num");
//        job.getCounters().findCounter("DISALLOWED_WITH_*", "num");
//        job.getCounters().findCounter("DISALLOWED_WITH_$", "num");
//
        return 0;
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = GetJobConf(args);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    Job GetJobConf(String[] args) throws IOException {
        Job job = Job.getInstance(getConf(), HW3.class.getCanonicalName());
        job.setJarByClass(HW3.class);

        job.setPartitionerClass(FirstPartitioner.class);
        job.setSortComparatorClass(KeyComparator.class);
        job.setGroupingComparatorClass(GroupComparator.class);

        String webpages_name = args[0];
        String websites_name = args[1];

        List<Scan> scans = new ArrayList<Scan>();
        Scan scan1 = new Scan();
        scan1.addColumn(Bytes.toBytes("docs"), Bytes.toBytes("url"));
        scan1.addColumn(Bytes.toBytes("docs"), Bytes.toBytes("disabled"));
        scan1.setAttribute(Scan.SCAN_ATTRIBUTES_TABLE_NAME, Bytes.toBytes(webpages_name));
        scans.add(scan1);

        Scan scan2 = new Scan();
        scan2.addColumn(Bytes.toBytes("info"), Bytes.toBytes("site"));
        scan2.addColumn(Bytes.toBytes("info"), Bytes.toBytes("robots"));
        scan2.setAttribute(Scan.SCAN_ATTRIBUTES_TABLE_NAME, Bytes.toBytes(websites_name));
        scans.add(scan2);

        TableMapReduceUtil.initTableMapperJob(
                scans,
                HW3Mapper.class,
                TextPair.class,
                Text.class,
                job
        );

        job.setMapOutputKeyClass(TextPair.class);
        job.setMapOutputValueClass(Text.class);

        TableMapReduceUtil.initTableReducerJob(
                webpages_name,
                HW3Reducer.class,
                job
        );



        return job;
    }

    static String getHost(String url) {
        String host;
        if (url.startsWith("http")) {
            host = url.split("/")[2];
        } else {
            host = url.split("/")[0];
        }

        return host;

    }

    static public class HW3Mapper extends TableMapper<TextPair, Text> {
        @Override
        protected void map(ImmutableBytesWritable rowKey, Result columns, Context context) throws IOException, InterruptedException {
            TableSplit currentSplit = (TableSplit) context.getInputSplit();
            String tableName = new String(currentSplit.getTableName());

            if (tableName.startsWith("websites")) {
                context.getCounter("websites", "num").increment(1);

                Cell robots_val = columns.getColumnLatestCell(Bytes.toBytes("info"), Bytes.toBytes("robots"));
                String robots = (robots_val != null) ? new String(CellUtil.cloneValue(robots_val), "UTF8") : "";

                Cell site_val = columns.getColumnLatestCell(Bytes.toBytes("info"), Bytes.toBytes("site"));
                String host = new String(CellUtil.cloneValue(site_val), "UTF8"); //new URL(new String(CellUtil.cloneValue(site_val), "UTF8"))).getHost();

                context.write(
                        new TextPair(new Text(host), new Text(new String("1"))),
                        new Text(robots)
                );
            } else if (tableName.startsWith("webpages")) {
                context.getCounter("webpages", "num").increment(1);

                Cell url_val = columns.getColumnLatestCell(Bytes.toBytes("docs"), Bytes.toBytes("url"));
                String url = new String(CellUtil.cloneValue(url_val), "UTF8");

                String host = getHost(url); // new URL(url)).getHost();

                Cell disabled_val = columns.getColumnLatestCell(Bytes.toBytes("docs"), Bytes.toBytes("disabled"));
                String disallowed_flag = (disabled_val != null) ? "Y" : "N";

                String mapper_value = Bytes.toString(rowKey.get()) + "\t" + (new URL(url)).getFile() + "\t" + disallowed_flag;

                context.write(
                        new TextPair(new Text(host), new Text(new String("2"))),
                        new Text(mapper_value)
                );
            }
        }
    }

    static public class HW3Reducer extends TableReducer<TextPair, Text, ImmutableBytesWritable> {
        @Override
        protected void reduce(TextPair key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String robots = values.iterator().next().toString();
            context.getCounter("NUMBER_OF_REDUCERS", "num").increment(1);

            List<Pair<String, String>> rules = new ArrayList<Pair<String, String>>();
            if (robots.startsWith("Disallow")) {
                context.getCounter("ROBOTS_COUNTER", "num").increment(1);

                String[] robots_rules = robots.split("\n");

                if (robots_rules.length > 1){
                    context.getCounter("OK_ROBOTS", "num").increment(1);
                }
                if (robots_rules.length == 0){
                    context.getCounter("BAD_ROBOTS", "num").increment(1);
                }

                // Disallow: /info (запрещает пути, начинающиеся с /info: /info, /info/help.html, ...)
                // Disallow: *forum (запрещает все пути, где есть слово forum: /forum, /attic/forum27/, ...)
                // Disallow: remove.php$ (запрещает все пути, оканчивающиеся на remove.php)

                for (String rule : robots_rules) {
                    String rule_str = rule.split(" ")[1].trim();

                    Pair<String, String> rulePair = new Pair<>();

                    if (rule_str.endsWith("$")) {
                        rulePair = new Pair<>("$", rule_str.substring(0, rule_str.length()-1));
                    }

                    if (rule_str.startsWith("/")) {
                        rulePair = new Pair<>("/", rule_str);
                    }

                    if (rule_str.startsWith("*")) {
                        rulePair = new Pair<>("*", rule_str.substring(1, rule_str.length()));
                    }

                    if (rule_str.startsWith("/") && rule_str.endsWith("$")) {
                        rulePair = new Pair<>("/$", rule_str.substring(0, rule_str.length()-1));
                    }

                    if (rule_str.startsWith("*") && rule_str.endsWith("$")) {
                        rulePair = new Pair<>("*$", rule_str.substring(1, rule_str.length()-1));
                    }

                    context.getCounter("ROBOTS_WITH_" + rulePair.getFirst(), "num").increment(1);

                    rules.add(rulePair);
                }
            }

            for (Text val : values) {
                context.getCounter("TOTAL_URLS", "num").increment(1);

                String[] page_meta = val.toString().split("\t");
                String row_key = page_meta[0];
                String url = page_meta[1];
                String disallowed_flag = page_meta[2];

                boolean is_disallowed = false;
                for (Pair<String, String> r : rules) {
                    if (r.getFirst().equals("/")) {
                        if (url.startsWith(r.getSecond())) {
                            context.getCounter("DISALLOWED_WITH_/", "num").increment(1);
                            is_disallowed = true;
                            break;
                        }
                    } else if (r.getFirst().equals("*")) {
                        if (url.contains(r.getSecond())) {
                            context.getCounter("DISALLOWED_WITH_*", "num").increment(1);
                            is_disallowed = true;
                            break;
                        }
                    } else if (r.getFirst().equals("$")) {
                        if (url.endsWith(r.getSecond())) {
                            context.getCounter("DISALLOWED_WITH_$", "num").increment(1);
                            is_disallowed = true;
                            break;
                        }
                    } else if (r.getFirst().equals("/$")) {
                        if (url.equals(r.getSecond())) {
                            context.getCounter("DISALLOWED_WITH_/$", "num").increment(1);
                            is_disallowed = true;
                            break;
                        }
                    } else if (r.getFirst().equals("*$")) {
                        if (url.endsWith(r.getSecond())) {
                            context.getCounter("DISALLOWED_WITH_*$", "num").increment(1);
                            is_disallowed = true;
                            break;
                        }
                    }
                }

                if (is_disallowed && !disallowed_flag.equals("Y")) {
                    Put put = new Put(Bytes.toBytes(row_key));
                    put.addColumn(Bytes.toBytes("docs"), Bytes.toBytes("disabled"), Bytes.toBytes("Y"));
                    context.getCounter("UPDATED_PAGES", "num").increment(1);
                    context.write(null, put);
                }

                if (!is_disallowed && disallowed_flag.equals("Y")) {
                    Delete del = new Delete(Bytes.toBytes(row_key));
                    del.addColumn(Bytes.toBytes("docs"), Bytes.toBytes("disabled"));
                    context.getCounter("UPDATED_PAGES", "num").increment(1);
                    context.write(null, del);
                }
            }
        }
    }

    public static class FirstPartitioner
            extends Partitioner<TextPair, NullWritable> {

        @Override
        public int getPartition(TextPair key, NullWritable value, int numPartitions) {
            return Math.abs(key.hashCode()) % numPartitions;
        }
    }

    public static class KeyComparator extends WritableComparator {
        protected KeyComparator() {
            super(TextPair.class, true);
        }

        @Override
        public int compare(WritableComparable w1, WritableComparable w2) {
            TextPair ip1 = (TextPair) w1;
            TextPair ip2 = (TextPair) w2;

            return ip1.compareTo(ip2);
        }
    }

    public static class GroupComparator extends WritableComparator {
        protected GroupComparator() {
            super(TextPair.class, true);
        }

        @Override
        public int compare(WritableComparable w1, WritableComparable w2) {
            TextPair ip1 = (TextPair) w1;
            TextPair ip2 = (TextPair) w2;
            return ip1.getFirst().compareTo(ip2.getFirst());
        }
    }

    public static void main(String[] args) throws Exception {
        int rc = ToolRunner.run(HBaseConfiguration.create(), new HW3(), args);
        System.exit(rc);
    }
}
