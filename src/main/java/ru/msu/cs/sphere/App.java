package ru.msu.cs.sphere;

import java.io.InputStream;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;


public class App {

    static public String[] TICKET_TYPES = {"взрослый", "детский", "льготный"};
    static public DateTimeFormatter formatter =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss Z");


    static class Interval {
        public LocalDate start;
        public LocalDate end;
        public Map<String, Integer> ticketsDist;

        public Interval(LocalDate inStart, LocalDate inEnd, Map<String, Integer> inTicketsDist) {
            start = inStart;
            end = inEnd;
            ticketsDist = inTicketsDist;
        }
    }

    static public class LogEntityComparator implements Comparator<LogEntity> {
        public int compare(LogEntity o1, LogEntity o2) {
            return o1.date.compareTo(o2.date);
        }
    }

    static class LogEntity {
        public LocalDate date;
        public String ticketType;
        public Boolean action;
        public String userId;

        public LogEntity() {
        }

        public LogEntity(LocalDate inDate, String inTicketType,
                         Boolean inAction, String inUserId) {
            date = inDate;
            ticketType = inTicketType;
            action = inAction;
            userId = inUserId;
        }
    }

    static class Reader {
        public List<LogEntity> getData(InputStream inp) {
            ArrayList<LogEntity> list = new ArrayList<LogEntity>();

            Scanner in = new Scanner(inp);
            while (in.hasNextLine()) {
                String next = in.nextLine();
                if (next.equals("")) break;

                String[] inArray = in.nextLine().split(";", 4);
                if (inArray.length != 4)
                    System.out.println("Bad args");

                LocalDate logEntityDate = LocalDate.parse(inArray[0], formatter);

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
        LocalDate currentDate = null;
        Integer previousVisitorsAmount = 0;
        LocalDate previousDate = null;

        Map<String, Integer> ticketsDist = getEmptyTicketsDist();

        Collections.sort(logEntities, new LogEntityComparator()); // -- ??? how?

        for (LogEntity log : logEntities) {
            Integer action = log.action ? 1 : -1;
            currentDate = log.date;

            currentVisitorsAmount += action;

            if (currentVisitorsAmount > maxVisitorsAmount) {
                maxVisitorsAmount = currentVisitorsAmount;
                popularIntervals.clear();
            }

            if (previousVisitorsAmount > currentVisitorsAmount) {
                if (previousVisitorsAmount.equals(maxVisitorsAmount)) {
                    Interval interval = new Interval(previousDate, currentDate, ticketsDist);
                    popularIntervals.add(interval);
                }
            }

            previousVisitorsAmount = currentVisitorsAmount;
            previousDate = currentDate;

            int count = ticketsDist.containsKey(log.ticketType) ? ticketsDist.get(log.ticketType) : 0;
            ticketsDist.put(log.ticketType, count + 1);
        }

        if (previousVisitorsAmount.equals(maxVisitorsAmount)) {
            Interval interval = new Interval(previousDate, null, ticketsDist);
            popularIntervals.add(interval);
        }

        return popularIntervals;
    }

    public static void main(String[] args) {
        Reader reader = new Reader();
        List<LogEntity> inArray = reader.getData(System.in);

        List<Interval> popularIntervals = computePopularTimeIntervals(inArray);

        for (Interval interval : popularIntervals) {
            System.out.println(interval.start.toString());
            System.out.println(interval.end.toString());
        }
    }

    static public int sayHello() {
        System.out.println("Hello!");
        return 1;
    }
}
