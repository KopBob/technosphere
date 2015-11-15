package ru.msu.cs.sphere;

import java.io.InputStream;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;


public class App {
    static public final String ADULT_TICKET_TYPE = "взрослый";
    static public final String CHILDISH_TICKET_TYPE = "детский";
    static public final String PREFERENTIAL_TICKET_TYPE = "льготный";

    static public final String DATE_FORMAT = "yyyy-MM-dd HH:mm:ss Z";

    static public String[] TICKET_TYPES = new String[]{
            ADULT_TICKET_TYPE,
            CHILDISH_TICKET_TYPE,
            PREFERENTIAL_TICKET_TYPE
    };

    static public SimpleDateFormat FORMATTER = new SimpleDateFormat(DATE_FORMAT);


    static class Interval {
        public Date start;
        public Date end;
        public Map<String, Integer> ticketsDist;

        public Interval(Date inStart, Date inEnd, Map<String, Integer> inTicketsDist) {
            start = inStart;
            end = inEnd;
            ticketsDist = new HashMap<String, Integer>(inTicketsDist); // ???
        }

        public String repr() {
            int visitorsAmount = 0;
            for (int num : ticketsDist.values())
                visitorsAmount = visitorsAmount + num;

            Float adultPortion =
                    ((float) ticketsDist.get(ADULT_TICKET_TYPE) / visitorsAmount) * 100;
            Float childishPortion =
                    ((float) ticketsDist.get(CHILDISH_TICKET_TYPE) / visitorsAmount) * 100;
            Float preferentialPortion =
                    ((float) ticketsDist.get(PREFERENTIAL_TICKET_TYPE) / visitorsAmount) * 100;

            String startStr = FORMATTER.format(this.start);
            String endStr = this.end != null ? FORMATTER.format(this.end) : "";

            return String.format("%s;%s;%.1f;%.1f;%.1f", startStr, endStr,
                    adultPortion, childishPortion, preferentialPortion);
        }

        public void print() {
            System.out.println(this.repr());
        }

        public void prettyPrint() {
            System.out.println(FORMATTER.format(this.start));
            System.out.println(FORMATTER.format(this.end));
            for (String type : TICKET_TYPES) {
                System.out.print(type);
                System.out.print("  ");
                System.out.println(this.ticketsDist.get(type));
            }

            System.out.println();
        }
    }

    static class LogEntity {
        public Date date;
        public String ticketType;
        public Boolean action;
        public String userId;

        public LogEntity() {}

        public LogEntity(Date inDate, String inTicketType,
                         Boolean inAction, String inUserId) {
            date = inDate;
            ticketType = inTicketType;
            action = inAction;
            userId = inUserId;
        }
    }

    static public class LogEntityComparator implements Comparator<LogEntity> {
        public int compare(LogEntity o1, LogEntity o2) {
            return o1.date.compareTo(o2.date);
        }
    }

    static class Reader {
        public List<LogEntity> getData(InputStream inp) throws ParseException {
            ArrayList<LogEntity> list = new ArrayList<LogEntity>();

            Scanner in = new Scanner(inp);

            while (in.hasNextLine()) {
                String next = in.nextLine();
                if (next.equals("")) continue;

                String[] inArray = next.split(";", 4);
                if (inArray.length != 4)
                    System.out.println("Bad args");

                Date logEntityDate = FORMATTER.parse(inArray[0]);

                if (!Arrays.asList(TICKET_TYPES).contains(inArray[1]))
                    System.out.println("Invalid Ticket Type!");

                String ticketType = inArray[1];
                Boolean logEntityAction = Byte.parseByte(inArray[2]) > 0;
                String userId = inArray[3];

                LogEntity logEntity = new LogEntity(logEntityDate, ticketType,
                        logEntityAction, userId);

                list.add(logEntity);
            }
            return list;
        }
    }

    public static Map<String, Integer> getEmptyTicketsDist() {
        Map<String, Integer> map = new HashMap<String, Integer>();
        for (String type : TICKET_TYPES) map.put(type, 0);
        return map;
    }

    public static List<Interval> computePopularTimeIntervals(List<LogEntity> logEntities) {
        List<Interval> popularIntervals = new ArrayList<Interval>();

        Integer maxVisitorsAmount = -1;

        Integer currentVisitorsAmount = 0;
        Date currentDate = null;
        Integer previousVisitorsAmount = 0;
        Date previousDate = null;

        Map<String, Integer> ticketsDist = getEmptyTicketsDist();

        Collections.sort(logEntities, new LogEntityComparator()); // -- ??? how?

        for (LogEntity currentLog : logEntities) {
            Integer action = currentLog.action ? 1 : -1;

            currentVisitorsAmount += action;

            if (currentVisitorsAmount > maxVisitorsAmount) {
                maxVisitorsAmount = currentVisitorsAmount;
                popularIntervals.clear();
            }

            if (previousVisitorsAmount > currentVisitorsAmount) {
                if (previousVisitorsAmount.equals(maxVisitorsAmount)) {
                    Interval interval = new Interval(previousDate, currentLog.date, ticketsDist);
                    popularIntervals.add(interval);
                }
            }

            previousVisitorsAmount = currentVisitorsAmount;
            previousDate = currentLog.date;

            ticketsDist.put(currentLog.ticketType, ticketsDist.get(currentLog.ticketType) + action);
        }

        if (previousVisitorsAmount.equals(maxVisitorsAmount)) {
            Interval interval = new Interval(previousDate, null, ticketsDist);
            popularIntervals.add(interval);
        }

        return popularIntervals;
    }


    public static void main(String[] args) throws ParseException {
        Reader reader = new Reader();

        List<LogEntity> inArray = reader.getData(System.in);
        List<Interval> popularIntervals = computePopularTimeIntervals(inArray);

        for (Interval interval : popularIntervals) {
            interval.print();
        }
    }
}
